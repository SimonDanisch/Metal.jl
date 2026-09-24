export @metal


## high-level @metal interface

const MACRO_KWARGS = [:launch]
const COMPILER_KWARGS = [:kernel, :name, :always_inline, :debug_level, :opt_level, :macos, :air, :metal, :gpufamily]
const LAUNCH_KWARGS = [:groups, :threads, :queue, :submit]

"""
    @metal threads=... groups=... [kwargs...] func(args...)

High-level interface for executing code on a GPU.

The `@metal` macro should prefix a call, with `func` a callable function or object that
should return nothing. It will be compiled to a Metal function upon first use, and to a
certain extent arguments will be converted and managed automatically using `mtlconvert`.
Finally, a call to `mtlcall` is performed, encoding the kernel onto the selected command
queue and submitting according to that queue's batching policy.

There are a few keyword arguments that influence the behavior of `@metal`:

- `launch`: whether to launch this kernel, defaults to `true`. If `false`, the returned
  kernel object should be launched by calling it and passing arguments again.
- `name`: the name of the kernel in the generated code. Defaults to an automatically-
  generated name.
- `opt_level`: the optimization level used when compiling the kernel, an integer from `0`
  to `3`. Defaults to `2`, independent of the host session's `-O` level.
- `queue`: the command queue to use for this kernel. Defaults to the global command queue.
- `submit`: whether to submit the current command batch immediately after encoding this
  kernel. Defaults to `false`.
"""
macro metal(ex...)
    call = ex[end]
    kwargs = map(ex[1:end-1]) do kwarg
        if kwarg isa Symbol
            :($kwarg = $kwarg)
        elseif Meta.isexpr(kwarg, :(=))
            kwarg
        else
            throw(ArgumentError("Invalid keyword argument '$kwarg'"))
        end
    end

    # destructure the kernel call
    Meta.isexpr(call, :call) || throw(ArgumentError("second argument to @metal should be a function call"))
    f = call.args[1]
    args = call.args[2:end]

    code = quote end
    vars, var_exprs = assign_args!(code, args)

    # group keyword argument
    macro_kwargs, compiler_kwargs, call_kwargs, other_kwargs =
        split_kwargs(kwargs, MACRO_KWARGS, COMPILER_KWARGS, LAUNCH_KWARGS)
    if !isempty(other_kwargs)
        key,_ = first(other_kwargs).args
        throw(ArgumentError("Unsupported keyword argument '$key'"))
    end

    # handle keyword arguments that influence the macro's behavior
    launch = true
    for kwarg in macro_kwargs
        key,val = kwarg.args
        if key === :launch
            isa(val, Bool) || throw(ArgumentError("`launch` keyword argument to @metal should be a Bool"))
            launch = val::Bool
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end
    if !launch && !isempty(call_kwargs)
        error("@metal with launch=false does not support launch-time keyword arguments; use them when calling the kernel")
    end

    # FIXME: macro hygiene wrt. escaping kwarg values (this broke with 1.5)
    #        we esc() the whole thing now, necessitating gensyms...
    @gensym f_var kernel_f kernel_args kernel_tt kernel

    # convert the arguments, call the compiler and launch the kernel
    # while keeping the original arguments alive
    push!(code.args,
        quote
            $f_var = $f
            GC.@preserve $(vars...) $f_var begin
                $kernel_f = $mtlconvert($f_var)
                $kernel_args = map($mtlconvert, ($(var_exprs...),))
                $kernel_tt = Tuple{map(Core.Typeof, $kernel_args)...}
                $kernel = $mtlfunction($kernel_f, $kernel_tt; $(compiler_kwargs...))
                if $launch
                    $kernel($(var_exprs...); $(call_kwargs...))
                end
                $kernel
            end
         end)

    return esc(quote
        let
            $code
        end
    end)
end


## argument conversion

struct Adaptor
    # the current command encoder, if any.
    cce::Union{Nothing,MTLComputeCommandEncoder}
end

"""
Make `buf` resident for every dispatch on its device's queue.

For the case where a GPU address is handed out with no encoder in scope — see
the `adapt_storage` below. Metal's residency set is the only mechanism that
works then: `useResource` needs an encoder, and by the time one exists nobody
remembers this buffer.

A buffer already in the set is skipped. This does NOT run only at scene setup, as
this comment used to claim: every launch that bakes an address arrives here again
with the same buffers, and the add-and-commit pair is a driver call each time. A
set never gives an allocation back, so "already added" is permanent and one
pointer lookup answers it — see `residency_members`.
"""
make_persistently_resident!(buf::MTLBuffer) = make_persistently_resident!(buf, buf.device)

# Any allocation, not only a buffer. An `MTLIndirectCommandBuffer` is reached by
# the command processor exactly the way a buffer is reached by a shader — by
# address, with nothing in the submission naming it — so on a path with no
# `useResource` (Metal 4) it has to be in the set or the replay faults. It does
# not carry a `device` property of its own, hence the explicit one.
function make_persistently_resident!(buf, dev::MTLDevice)
    can_use_residency_sets(dev) || return buf
    bq = global_queue(dev)
    queue = bq isa MTLCommandQueue ? bq : getfield(bq, :queue)
    resset = install_queue_residency!(queue, dev)
    known = Base.@lock queue_residency_sets_lock begin
        members = get!(Set{UInt}, residency_members, UInt(pointer(resset)))
        UInt(pointer(buf)) in members ? true : (push!(members, UInt(pointer(buf))); false)
    end
    known && return buf
    MTL.add_allocation!(resset, buf)
    MTL.commit!(resset)
    return buf
end

# convert Metal buffers to their GPU address
function Adapt.adapt_storage(to::Adaptor, buf::MTLBuffer)
    if to.cce !== nothing
        # Inside a launch: the encoder makes it resident for this dispatch.
        MTL.use!(to.cce, buf, MTL.ReadWriteUsage)
    else
        # No encoder. The caller is converting an array to a raw GPU address to
        # STORE somewhere — a pointer table, an argument buffer, a struct field
        # the launch path never walks. Nothing will `useResource` it at dispatch
        # time, so unless it is made resident now the GPU reads unmapped memory.
        #
        # That failure is silent and intermittent: reads come back as zeros when
        # the page happens not to be mapped, which is why it shows up as a
        # texture that is sometimes black. Raycore's `store_texture` reaches
        # exactly this path via `KA.argconvert`, and its image textures and
        # environment maps were the symptom.
        make_persistently_resident!(buf)
    end
    reinterpret(Core.LLVMPtr{Nothing,AS.Device}, buf.gpuAddress)
end
function Adapt.adapt_storage(to::Adaptor, ptr::MtlPtr{T}) where {T}
    reinterpret(Core.LLVMPtr{T,AS.Device}, adapt(to, ptr.buffer)) + ptr.offset
end

# convert Metal host arrays to device arrays
function Adapt.adapt_storage(to::Adaptor, xs::MtlArray{T,N}) where {T,N}
    buf = pointer(xs)
    ptr = adapt(to, buf)
    MtlDeviceArray{T,N,AS.Device}(xs.dims, ptr)
end

# Base.RefValue isn't GPU compatible, so provide a compatible alternative
# TODO: port improvements from CUDA.jl
struct MtlRefValue{T} <: Ref{T}
    x::T
end
Base.getindex(r::MtlRefValue) = r.x
Adapt.adapt_structure(to::Adaptor, r::Base.RefValue) = MtlRefValue(adapt(to, r[]))

# broadcast sometimes passes a ref(type), resulting in a GPU-incompatible DataType box.
# avoid that by using a special kind of ref that knows about the boxed type.
struct MtlRefType{T} <: Ref{DataType} end
Base.getindex(::MtlRefType{T}) where {T} = T
Adapt.adapt_structure(::Adaptor, r::Base.RefValue{<:Union{DataType, Type}}) =
    MtlRefType{r[]}()

# case where type is the function being broadcasted
Adapt.adapt_structure(to::Adaptor,
                      bc::Broadcast.Broadcasted{Style, <:Any, Type{T}}) where {Style, T} =
    Broadcast.Broadcasted{Style}((x...) -> T(x...), adapt(to, bc.args), bc.axes)

"""
    mtlconvert(x, [cce])

This function is called for every argument to be passed to a kernel, allowing it to be
converted to a GPU-friendly format. By default, the function does nothing and returns the
input object `x` as-is.

Do not add methods to this function, but instead extend the underlying Adapt.jl package and
register methods for the the `Metal.Adaptor` type.
"""
mtlconvert(arg, cce=nothing) = adapt(Adaptor(cce), arg)


## host-side kernel API

struct HostKernel{F,TT}
    f::F
    pipeline::MTLComputePipelineState
    loggingEnabled::Bool
    device::MTLDevice
    maxthreads::Int
    tgmem::Int
    exec_width::Int
    use_residency_sets::Bool
    # this session's relocation words, or `nothing` for a relocation-free kernel. The buffer
    # keeps the storage alive; `launch` passes its address in the `KernelState` and declares
    # it resident, since only the address (not the buffer) is encoded.
    reloc_table::Union{Nothing,MTLBuffer}
end

const mtlfunction_lock = ReentrantLock()

"""
    mtlfunction(f, tt=Tuple{}; kwargs...)

Low-level interface to compile a function invocation for the currently-active GPU, returning
a callable kernel object. For a higher-level interface, use [`@metal`](@ref).

The following keyword arguments are supported:
- `macos`, `metal` and `air`: to override the macOS OS, Metal language and AIR bitcode
   versions used during compilation. Value should be a valid version number.
- `gpufamily`: to override the Apple GPU family (`MTL.MTLGPUFamilyApple<n>`) that the
   generated code may rely on. Defaults to the highest family the device supports.
- `indirect`: build a pipeline an `MTLIndirectCommandBuffer` command may name. The
   compiled code is the same and is cached the same; only the pipeline differs, and
   it is not cached, because a caller who asks for one holds it.

The output of this function is automatically cached, i.e. you can simply call `mtlfunction`
in a hot path without degrading performance. New code will be generated automatically when
the function changes, or when different types or keyword arguments are provided.
"""
function mtlfunction(f::F, tt::TT=Tuple{}; name=nothing, indirect::Bool=false,
                     kwargs...) where {F,TT}
    Base.@lock mtlfunction_lock begin
        dev = device()
        config = compiler_config(dev; name, kwargs...)::MetalCompilerConfig
        source = methodinstance(F, tt)
        job = CompilerJob(source, config)

        res = compile_or_lookup(job)::MetalResults

        # Resolve the MTLComputePipelineState for the active device. Linear scan
        # over the session-local cache; almost always n=1, one `===` compare.
        #
        # An `indirect` pipeline — built with `supportIndirectCommandBuffers` — is a
        # different object from the one a `@metal` launch wants, so it has its own
        # list. It used to be built afresh on every call, on the reasoning that the
        # caller holds it; but Mantle's recorder asks again every time a plan is
        # REBUILT, and a RayMakie material switch rebuilds the renderer's plans.
        # Measured on the isubd demo: 136 pipelines re-linked from code that was
        # already compiled, 95 s of a 147 s stall, each one a native compile of
        # the linked traversal. A pipeline state is immutable and may be named by
        # any number of commands, so sharing it is safe.
        pipes = indirect ? res.indirect_pipelines : res.pipelines
        pipeline = Ref{MTLComputePipelineState}()
        @inbounds for (cached_dev, cached_pipeline) in pipes
            if cached_dev === dev
                pipeline[] = cached_pipeline
                break
            end
        end
        if !isassigned(pipeline)
            pipeline[] = link_pipeline(dev, res.air::Vector{UInt8},
                                     res.metallib::Vector{UInt8},
                                     res.entry::String; indirect)
            # Don't cache session-local pipeline handles while precompiling: the
            # results struct is serialized into the package image along with its
            # CodeInstance, and ObjectiveC handles would come back dangling.
            if ccall(:jl_generating_output, Cint, ()) != 1
                push!(pipes, (dev, pipeline[]))
            end
        end

        # Same for the relocation words, which only a kernel referencing a host object needs.
        relocations = res.relocations
        reloc_table = nothing
        if relocations !== nothing
            @inbounds for (cached_dev, cached_buf) in res.reloc_tables
                if cached_dev === dev
                    reloc_table = cached_buf
                    break
                end
            end
            if reloc_table === nothing
                reloc_table = reloc_table_buffer(dev, relocations)
                if ccall(:jl_generating_output, Cint, ()) != 1
                    push!(res.reloc_tables, (dev, reloc_table))
                end
            end
        end

        h = hash(pipeline[], hash(f, hash(tt)))
        get!(kernel_instances, h) do
            local dev = pipeline[].device
            HostKernel{F,tt}(f, pipeline[], res.loggingEnabled::Bool,
                             dev,
                             Int(pipeline[].maxTotalThreadsPerThreadgroup),
                             Int(pipeline[].staticThreadgroupMemoryLength),
                             Int(pipeline[].threadExecutionWidth),
                             can_use_residency_sets(dev),
                             reloc_table)
        end::HostKernel{F,tt}
    end
end

# Look up cached compile artifacts for `job`, compiling on miss. Storage is managed
# by `GPUCompiler.cached_results` (Julia's integrated code cache on 1.11+, which also
# persists artifacts through precompilation; a session-local store on 1.10).
#
# `metallib === nothing` identifies a `MetalResults` that hasn't been compiled yet —
# either freshly created, or (on 1.11+) loaded from a package image whose precompile
# workload only inferred the kernel without compiling it. The `compile_hook` check
# additionally forces the compile path so reflection-style consumers (`@device_code_*`)
# observe the compilation even on a cache hit.
# How often a kernel was served from the compile cache, and how often it had to be
# built. The same two numbers Lava reports for its frozen SPIR-V, so that Mantle's
# `kernelcompiles` reads alike on both backends rather than each inventing a shape.
#
# `Ref` and not `Threads.Atomic`, matching Lava: these are diagnostics, and a lost
# increment under contention costs a count, not correctness.
const COMPILE_HITS = Ref(0)
const COMPILE_MISSES = Ref(0)

"""
    compile_stats() -> (; hits, misses)

Kernels served from the compile cache, and kernels built because they were not in
it. Read through `Mantle.kernelcompiles(device)`; reset with
[`reset_compile_stats!`](@ref).
"""
compile_stats() = (; hits = COMPILE_HITS[], misses = COMPILE_MISSES[])

"""Zero both counters, so a measured region starts from a known point."""
reset_compile_stats!() = (COMPILE_HITS[] = 0; COMPILE_MISSES[] = 0; nothing)

# Specialize on the target/parameter types so callers can avoid boxing CompilerJob.
# Keep the body out of callers that specialize per kernel. (Upstream #967.)
@noinline function compile_or_lookup(job::CompilerJob)::MetalResults
    res = GPUCompiler.cached_results(MetalResults, job)
    if res === nothing || res.metallib === nothing || GPUCompiler.compile_hook[] !== nothing
        COMPILE_MISSES[] += 1
        artifacts = compile_to_metallib(job)
        res = @something res GPUCompiler.cached_results(MetalResults, job)
        res.air = artifacts.air
        res.metallib = artifacts.metallib
        res.entry = artifacts.entry
        res.loggingEnabled = artifacts.loggingEnabled
        res.relocations = artifacts.relocations
    else
        COMPILE_HITS[] += 1
    end
    return res
end

# cache of kernel instances
const kernel_instances = Dict{UInt, Any}()


## kernel launching and argument encoding

# `args::Tuple` and not `Vararg`: splatting one to reach this built a NEW tuple on
# every launch — 1176 of 8256 sampled bytes in a 400-launch profile, attributed to the
# splat in `encode_arguments_nospec!`. A generated function reads the field types of a
# tuple exactly as it reads a vararg's, so nothing about the generated code changes.
@inline @generated function encode_arguments!(cce, kernel, kernel_state, f, args::Tuple)
    ex = quote end

    # the arguments passed into this function have not been `mtlconvert`ed, because we need
    # to retain the top-level MTLBuffer and MtlPtr objects. eager conversion of nested
    # such objects to LLVMPtr seems fine, somehow.
    # TODO: can we just convert everything eagerly and support top-level LLVMPtrs?

    # The kernel state and the function come first and by name; everything after them
    # is read out of the argument TUPLE. Splicing all three into one tuple to iterate
    # uniformly is what the caller used to do, and building it allocated on every
    # launch — 2872 of 7432 sampled bytes in a 400-launch profile, the largest site
    # left. Here nothing is spliced: `kernel_state` and `f` are parameters.
    idx = 1
    for (argidx, argtyp) in enumerate((kernel_state, f, fieldtypes(args)...))
        argex = argidx == 1 ? :(kernel_state) :
                argidx == 2 ? :(f) : :(args[$(argidx - 2)])
        if argtyp <: MTLBuffer
            # top-level buffers are passed as a pointer-valued argument
            push!(ex.args, :(set_buffer!(cce, $argex, 0, $idx)))
        elseif argtyp <: MtlPtr
            # the same as a buffer, but with an offset
            push!(ex.args, :(set_buffer!(cce, $argex.buffer, $argex.offset, $idx)))
        elseif isghosttype(argtyp) || Core.Compiler.isconstType(argtyp)
            continue
        else
            # everything else is passed by reference, copied into Metal's transient buffer
            append!(ex.args, (quote
                set_argument!(cce, mtlconvert($(argex), cce), $idx)
            end).args)
        end
        idx += 1
    end

    push!(ex.args, :(return nothing))

    ex
end

@inline function set_argument!(cce::MTLComputeCommandEncoder, arg, idx::Integer)
    argtyp = typeof(arg)

    # A non-isbits argument has no fields the kernel could read — compilation would have
    # failed otherwise — so it is only usable by identity, e.g. `sym === :foo`. Pass the
    # object's address, which is the identity word GPUCompiler lowers such a parameter to,
    # and which the kernel compares against its own resolved relocation for that value.
    if !isbitstype(argtyp)
        arg = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), arg)
        argtyp = Ptr{Cvoid}
    end

    # A `Ref` and not a scratch buffer owned by the queue. That was tried — a
    # `Vector{UInt8}` field on `BatchedCommandQueue`, threaded through the generated
    # encoder — on the theory that this allocates once per ARGUMENT per launch. It does
    # not: the optimiser already elides a `Ref` that `set_bytes!` only reads, and the
    # scratch measured 480 bytes a launch against 480 for this — no difference at all,
    # for a struct field and three signatures of plumbing. Reverted.
    ref = Base.RefValue(arg)
    GC.@preserve ref begin
        ptr = Base.unsafe_convert(Ptr{argtyp}, ref)
        set_bytes!(cce, reinterpret(Ptr{Cvoid}, ptr), sizeof(argtyp), idx)
    end
    return
end

# wraps a single function call, keeping its closure body small.
@autoreleasepool function (kernel::HostKernel)(args...; groups=1, threads=1,
                                               queue=nothing, submit::Bool=false,
                                               indirect=nothing)
    # function barrier to avoid capturing the `@autoreleasepool` in the generated code
    launch_with_queue(kernel, queue, MTLSize(groups), MTLSize(threads), args, submit,
                      indirect)
end

@inline function launch_with_queue(kernel::HostKernel, ::Nothing,
                                   gs::MTLSize, ts::MTLSize, args::Tuple,
                                   submit::Bool, indirect = nothing)
    launch(kernel, gs, ts, global_queue(device()), args, submit, indirect)
end

@inline function launch_with_queue(kernel::HostKernel, queue,
                                   gs::MTLSize, ts::MTLSize, args::Tuple,
                                   submit::Bool, indirect = nothing)
    launch(kernel, gs, ts, batched_queue(queue), args, submit, indirect)
end

function kernel_operation(kernel::HostKernel, gs::MTLSize, ts::MTLSize)
    (; kind = :kernel, name = string(nameof(kernel.f)),
       threadgroups = gs, threads = ts,
       tgmem = kernel.tgmem, maxthreads = kernel.maxthreads)
end

function launch_logging!(kernel::HostKernel, gs::MTLSize, ts::MTLSize,
                         bq::BatchedCommandQueue, @nospecialize(args::Tuple),
                         kernel_state, buf, exc)
    flush!(bq)
    queue = bq.queue

    is_macos(v"15") ||
        error("Capturing GPU log output requires macOS 15 or higher.")

    if is_virtual(queue.device)
        # `MTLLogState` needs a residency set, which the paravirtualized GPU driver
        # cannot create (failing with `MTLLogStateErrorDomain` code 2). Bail out here
        # with a clear host error instead of surfacing that opaque `NSError`.
        error("Capturing GPU log output is not supported on virtualized GPUs.")
    end

    MTLCaptureManager().isCapturing &&
        error("Logging is not supported while GPU frame capturing")

    log_state_descriptor = MTLLogStateDescriptor()
    log_state_descriptor.level = MTL.MTLLogLevelDebug
    log_state = MTLLogState(queue.device, log_state_descriptor)

    function log_handler(subSystem, category, logLevel, message)
        Core.print(String(NSString(message)))
        return nothing
    end

    block = @objcblock(log_handler, Nothing, (id{NSString}, id{NSString}, NSInteger, id{NSString}))
    @objc [log_state::id{MTLLogState} addLogHandler:block::id{NSBlock}]::Nothing

    cmdbuf_descriptor = MTLCommandBufferDescriptor()
    cmdbuf_descriptor.logState = log_state
    cmdbuf = MTLCommandBuffer(queue, cmdbuf_descriptor)
    @label! cmdbuf "MTLCommandBuffer($(nameof(kernel.f)))"
    let md = MTL.profile_metadata[]
        md === nothing || MTL.note_operation!(md, cmdbuf, kernel_operation(kernel, gs, ts))
    end

    cce = MTLComputeCommandEncoder(cmdbuf)
    try
        MTL.set_function!(cce, kernel.pipeline)
        if !kernel.use_residency_sets
            # DROP-MACOS14: per-launch residency for macOS 14 / virtual GPUs.
            MTL.use!(cce, buf, MTL.ReadWriteUsage)
            MTL.use!(cce, exc, MTL.ReadWriteUsage)
        end
        let reloc = kernel.reloc_table
            reloc === nothing || MTL.use!(cce, reloc, MTL.ReadUsage)
        end
        encode_arguments_nospec!(cce, kernel, kernel_state, kernel.f, args)
        MTL.append_current_function!(cce, gs, ts)
    finally
        close(cce)
    end

    commit!(cmdbuf, queue)
    defer_cleanup!(bq, cmdbuf, Any[kernel.f, args])
    track_logging_cmdbuf!(queue, cmdbuf)
    return
end

function launch(kernel::HostKernel, gs::MTLSize, ts::MTLSize,
                bq::BatchedCommandQueue, args::Tuple, submit::Bool,
                indirect = nothing)
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    (gs.width>0 && gs.height>0 && gs.depth>0) ||
        throw(ArgumentError("All group dimensions should be non-zero"))
    (ts.width>0 && ts.height>0 && ts.depth>0) ||
        throw(ArgumentError("All thread dimensions should be non-zero"))

    maxthreads = kernel.maxthreads
    nthreads = ts.width * ts.height * ts.depth
    nthreads > maxthreads &&
        throw(ArgumentError("Number of threads in group ($nthreads) should not exceed $maxthreads"))

    (gs.width * ts.width) > typemax(UInt32) &&
        throw(ArgumentError("Total threads per grid in a dimension (threads.width($(gs.width)) * groups.width($(ts.width)) = $(gs.width * ts.width)) must not exceed $(typemax(UInt32))"))
    (gs.height * ts.height) > typemax(UInt32) &&
        throw(ArgumentError("Total threads per grid in a dimension (threads.height($(gs.height)) * groups.height($(ts.height)) = $(gs.height * ts.height)) must not exceed $(typemax(UInt32))"))
    (gs.depth * ts.depth) > typemax(UInt32) &&
        throw(ArgumentError("Total threads per grid in a dimension (threads.depth($(gs.depth)) * groups.depth($(ts.depth)) = $(gs.depth * ts.depth)) must not exceed $(typemax(UInt32))"))

    f = kernel.f
    pipeline = kernel.pipeline
    dev = kernel.device
    tgmem = kernel.tgmem

    tgmem > 32768 &&
        throw(ArgumentError("Total used threadgroupMemoryLength($tgmem) must be <= 32768 bytes."))

    buf, buf_addr = malloc_buffer_and_gpu_address(dev)
    buf_ptr = reinterpret(Core.LLVMPtr{UInt8, AS.Device}, buf_addr)
    exc, exc_addr = exception_info_buffer_and_gpu_address(dev)
    exc_ptr = reinterpret(Core.LLVMPtr{UInt8, AS.Device}, exc_addr)
    reloc = kernel.reloc_table
    reloc_ptr = reinterpret(Core.LLVMPtr{UInt64, AS.Device},
                            reloc === nothing ? UInt64(0) : UInt64(reloc.gpuAddress))
    kernel_state = KernelState(Random.rand(UInt32), buf_ptr, exc_ptr, reloc_ptr)

    if kernel.loggingEnabled
        precompiling && return
        launch_logging!(kernel, gs, ts, bq, args, kernel_state, buf, exc)
        return
    end

    try
        cce = compute_encoder(bq)
        set_pipeline!(bq, cce, pipeline)

        # The kernel state holds GPU addresses to per-device scratch buffers (malloc bump
        # allocator, exception mailbox) that aren't otherwise bound to the encoder. Declare
        # them so Metal Shader Validation tracks the accesses instead of dropping them.
        if !kernel.use_residency_sets
            # DROP-MACOS14: per-launch residency for macOS 14 / virtual GPUs.
            MTL.use!(cce, buf, MTL.ReadWriteUsage)
            MTL.use!(cce, exc, MTL.ReadWriteUsage)
        end
        # The relocation table is per-kernel, so it cannot join the queue's residency set
        # (which only holds the per-device scratch buffers): declare it every launch.
        reloc === nothing || MTL.use!(cce, reloc, MTL.ReadUsage)

        encode_arguments_nospec!(cce, kernel, kernel_state, f, args)
        # `indirect` is `(buffer, byte offset)` holding three `UInt32` threadgroup
        # counts the DEVICE wrote. `gs` is then a bound the caller supplied for its
        # own bookkeeping and the driver reads the real size at execution — which is
        # the whole point: nobody on the host ever learns the count, so nobody has to
        # wait for the kernel that produced it.
        if indirect === nothing
            MTL.append_current_function!(cce, gs, ts)
        else
            # The buffer is read by the command processor rather than by the shader,
            # so it is not covered by the argument encoding above and has to be made
            # resident explicitly.
            MTL.use!(cce, indirect[1], MTL.ReadUsage)
            MTL.dispatchThreadgroupsIndirect!(cce, indirect[1], indirect[2], ts)
        end
    catch
        # The failing launch has not been recorded yet. Keep any earlier
        # operations in this batch, but close encoder state dirtied by the
        # failed encode and drop an otherwise empty command buffer.
        end_encoder!(bq)
        if bq.nops == 0
            cmdbuf = bq.cmdbuf
            cmdbuf === nothing || discard_open_cmdbuf!(bq, cmdbuf)
        end
        rethrow()
    end

    # The command buffer retains explicitly encoded buffers, but that doesn't keep other
    # resources alive for which we've encoded the GPU address ourselves.
    op = MTL.profile_metadata[] === nothing ? nothing : kernel_operation(kernel, gs, ts)
    record_operation!(bq, f, args, op)

    if precompiling
        cmdbuf = bq.cmdbuf
        end_encoder!(bq)
        cmdbuf === nothing || discard_open_cmdbuf!(bq, cmdbuf)
        return
    end

    submit ? flush!(bq) : maybe_autoflush!(bq)
    return
end

# Force specialization on f, args AND the kernel.
#
# This only buys anything if the CALLER has them concretely: `launch` used to declare
# `@nospecialize(args::Tuple)`, which made every call here a runtime dispatch, and a
# runtime dispatch BOXES the isbits arguments it passes — `kernel_state` and the
# varargs. Measured on this exact call: 0 bytes when the tuple's type is known against
# 128 for one argument and 176 for four, per launch, which is what a still Hikari frame
# was paying hundreds of times over. `launch` now takes `args::Tuple` unannotated; the
# kernel stays `@nospecialize`d, which is where the compile-time saving actually is.
#
# A/B on one still Hikari frame, 250 warm samples then the best of 5 x 50:
# `@nospecialize(args)` 5.2 ms and 549200 B per sample, without it 4.4 ms and 536224 B.
# The specialization is not just cheaper to run, it is cheaper to launch.
#
# `launch` and `launch_with_queue` dropped `@nospecialize(kernel::HostKernel)` for the
# same reason. A `@nospecialize`d struct is read through a dynamic `getfield`, so every
# `kernel.maxthreads`, `kernel.tgmem` and `kernel.loggingEnabled` in `launch` BOXED its
# `Int` or `Bool` — several per launch, and they are the sites a 400-launch profile
# attributed to `launch` itself. `HostKernel{F,TT}` is one type per kernel and the
# generated encoder already specializes per argument list, so this partitions nothing
# further than what was already there.
#
# The KERNEL is specialized on too, for the same reason. `encode_arguments!` is
# `@generated`, so it specializes on every argument type INCLUDING the kernel's;
# reaching it with a `@nospecialize`d kernel made that call dynamic as well, and a
# dynamic call boxes the isbits `KernelState` and the varargs it passes — measured
# 176 bytes for a four-argument kernel, per launch, against 0 once the type is
# known. Nothing is saved by hiding it: `HostKernel{F,TT}` is already one type per
# kernel, and the generated encoder was going to specialize per argument list
# anyway, which is the same partition.
@inline encode_arguments_nospec!(cce, kernel, kernel_state, f, args::Tuple) =
    encode_arguments!(cce, kernel, kernel_state, f, args)

## Intra-warp Helpers

"""
    nextwarp(dev, threads)
    prevwarp(dev, threads)

Returns the next or previous nearest number of threads that is a multiple of the warp size
of a device `dev`. This is a common requirement when using intra-warp communication.
"""
function nextwarp(pipe::MTLComputePipelineState, threads::Integer)
    ws = pipe.threadExecutionWidth
    return threads + (ws - threads % ws) % ws
end

function nextwarp(kernel::HostKernel, threads::Integer)
    ws = kernel.exec_width
    return threads + (ws - threads % ws) % ws
end

@doc (@doc nextwarp) function prevwarp(pipe::MTLComputePipelineState, threads::Integer)
    ws = pipe.threadExecutionWidth
    return threads - Base.rem(threads, ws)
end

@doc (@doc nextwarp) function prevwarp(kernel::HostKernel, threads::Integer)
    ws = kernel.exec_width
    return threads - Base.rem(threads, ws)
end
