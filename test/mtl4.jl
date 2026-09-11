# Metal 4, and what a migration to it would actually involve.
#
# Nothing in Metal.jl runs on Metal 4 today. This file exists because the day
# something forces the move — the MTL4 compiler, PSO specialisation, late-bound
# render targets, placement sparse, the machine-learning encoder, all of which are
# unreachable from an `MTLCommandQueue` — the first two days of that work are
# finding out which selectors exist and in what order they have to be called. That
# is done here, and it is asserted, so it stays true.
#
# ── What the runtime says the model is ───────────────────────────────────────
#
# `MTL4ComputeCommandEncoder` has NO `setBytes:` and NO `setBuffer:`: arguments are
# GPU ADDRESSES in an `MTL4ArgumentTable`. It has no `useResource:` either:
# residency is residency sets only. And it has no automatic hazard tracking, so
# every dependency the driver infers today becomes an explicit barrier.
#
# Those three are the migration: Metal.jl's `encode_arguments!` builds argument
# tables instead of pushing bytes, `make_persistently_resident!` becomes the only
# residency path, and the KA launch path grows barriers. Mantle's recorded plans
# need none of it — a recorded command already binds buffer slices by address,
# which is the MTL4 model.
#
# ── What it is NOT ───────────────────────────────────────────────────────────
#
# It is not faster. Measured 2026-09-10 on an M5, interleaved (six batches of 150
# submits, each batch's minimum, because a single before/after pass drifts by more
# than the effect): a one-segment ICB replay is 4.46 us on the legacy queue against
# 5.08 us on MTL4, and a one-kernel launch 5.29 against 5.92. A first,
# non-interleaved pass read the opposite and looked like a 2.3 us win for MTL4.
# Do not quote a Metal host-time comparison that was not interleaved.

using Test, Metal

const MTL4_SELS = Dict(
    "MTL4CommandQueue"          => ["commit:count:", "addResidencySet:", "signalEvent:value:"],
    "MTL4CommandAllocator"      => ["reset"],
    "MTL4CommandBuffer"         => ["beginCommandBufferWithAllocator:", "computeCommandEncoder",
                                    "endCommandBuffer", "useResidencySet:"],
    "MTL4ComputeCommandEncoder" => ["setArgumentTable:", "setComputePipelineState:",
                                    "dispatchThreadgroups:threadsPerThreadgroup:",
                                    "executeCommandsInBuffer:withRange:",
                                    "executeCommandsInBuffer:indirectBuffer:"],
    "MTL4ArgumentTable"         => ["setAddress:atIndex:", "setResource:atBufferIndex:"])

struct ObjCMethodDesc
    name::Ptr{Cvoid}
    types::Cstring
end

"""Every selector a protocol declares, required or not, instance or class."""
function protocol_selectors(name::AbstractString)
    p = ccall(:objc_getProtocol, Ptr{Cvoid}, (Cstring,), name)
    p == C_NULL && return nothing
    out = String[]
    for req in (true, false), inst in (true, false)
        n = Ref{Cuint}(0)
        list = ccall(:protocol_copyMethodDescriptionList, Ptr{ObjCMethodDesc},
                     (Ptr{Cvoid}, Bool, Bool, Ptr{Cuint}), p, req, inst, n)
        list == C_NULL && continue
        for i in 1:n[]
            push!(out, unsafe_string(ccall(:sel_getName, Cstring, (Ptr{Cvoid},),
                                           unsafe_load(list, i).name)))
        end
    end
    return unique(out)
end

sel(name) = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), name)
rawptr(x) = reinterpret(Ptr{Cvoid}, pointer(x))
msg0(obj, s) = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), obj, sel(s))
supports(dev, s) = ccall(:objc_msgSend, Bool, (Ptr{Cvoid}, Ptr{Cvoid}), rawptr(dev), sel(s))

function mtl4_probe!(a::Metal.MtlVector{Float32}, value::Float32, useicb::Bool)
    dev = Metal.device()
    d = rawptr(dev)
    kernel = Metal.mtlfunction(mtl4_add!, Tuple{Metal.MtlDeviceVector{Float32,1}, Float32};
                               name = "mtl4_add", indirect = useicb)

    # Arguments live in a buffer, because MTL4 has no `setBytes`. The layout is the
    # one a launch pushes: kernel state at slot 1, then one slot per argument.
    argbuf = MTL.MTLBuffer(dev, 1024; storage = Metal.SharedStorage)
    base = convert(Ptr{UInt8}, MTL.contents(argbuf))
    _, maddr = Metal.malloc_buffer_and_gpu_address(dev)
    _, eaddr = Metal.exception_info_buffer_and_gpu_address(dev)
    state = Metal.KernelState(UInt32(1),
        reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, maddr),
        reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, eaddr),
        reinterpret(Core.LLVMPtr{UInt64, Metal.AS.Device}, UInt64(0)))
    function put(x, off)
        r = Ref(x)
        GC.@preserve r unsafe_copyto!(base + off,
            convert(Ptr{UInt8}, Base.unsafe_convert(Ptr{typeof(x)}, r)), sizeof(x))
    end
    put(state, 0); put(Metal.mtlconvert(a), 256); put(value, 512)

    # Residency is residency SETS here, so the set the ordinary queue already keeps
    # is the one to hand over — plus whatever this probe made itself.
    legacy = Metal.global_queue(dev)
    lq = legacy isa MTL.MTLCommandQueue ? legacy : getfield(legacy, :queue)
    resset = Metal.install_queue_residency!(lq, dev)

    icb = nothing
    if useicb
        desc = MTL.MTLIndirectCommandBufferDescriptor(; max_kernel_buffers = 3, ray_tracing = true)
        icb = MTL.MTLIndirectCommandBuffer(dev, desc, 1)
        c = MTL.indirect_compute_command(icb, 1)
        MTL.set_pipeline!(c, kernel.pipeline)
        for (i, off) in enumerate((0, 256, 512))
            MTL.set_kernel_buffer!(c, argbuf, off, i)
        end
        MTL.dispatch_threadgroups!(c, Metal.MTLSize(1), Metal.MTLSize(64))
        MTL.add_allocation!(resset, icb)
    end
    MTL.add_allocation!(resset, argbuf)
    MTL.add_allocation!(resset, a.data[])
    MTL.commit!(resset)

    q4 = msg0(d, "newMTL4CommandQueue")
    allocator = msg0(d, "newCommandAllocator")
    cb4 = msg0(d, "newCommandBuffer")
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
          q4, sel("addResidencySet:"), rawptr(resset))

    # The argument table replaces every `setBuffer`/`setBytes` a launch would do.
    atd = msg0(msg0(ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "MTL4ArgumentTableDescriptor"),
                    "alloc"), "init")
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, UInt64), atd, sel("setMaxBufferBindCount:"), 4)
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Bool), atd, sel("setInitializeBindings:"), true)
    err = Ref{Ptr{Cvoid}}(C_NULL)
    table = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
                  d, sel("newArgumentTableWithDescriptor:error:"), atd, err)
    gpu = UInt64(argbuf.gpuAddress)
    for (i, off) in enumerate((0, 256, 512))
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, UInt64, Csize_t),
              table, sel("setAddress:atIndex:"), gpu + UInt64(off), i - 1)
    end

    # One frame. The command buffer and the allocator are REUSED, which is the one
    # thing MTL4 gives that the legacy queue cannot.
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), allocator, sel("reset"))
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
          cb4, sel("beginCommandBufferWithAllocator:"), allocator)
    enc = msg0(cb4, "computeCommandEncoder")
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
          cb4, sel("useResidencySet:"), rawptr(resset))
    if useicb
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, MTL.NSRange),
              enc, sel("executeCommandsInBuffer:withRange:"), rawptr(icb), MTL.NSRange(0, 1))
    else
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), enc, sel("setArgumentTable:"), table)
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
              enc, sel("setComputePipelineState:"), rawptr(kernel.pipeline))
        ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Metal.MTLSize, Metal.MTLSize),
              enc, sel("dispatchThreadgroups:threadsPerThreadgroup:"), Metal.MTLSize(1), Metal.MTLSize(64))
    end
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), enc, sel("endEncoding"))
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), cb4, sel("endCommandBuffer"))
    cbs = Ref(cb4)
    GC.@preserve cbs ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}, Csize_t),
                           q4, sel("commit:count:"), Base.unsafe_convert(Ptr{Ptr{Cvoid}}, cbs), 1)

    # Completion is an event: there is no `waitUntilCompleted` on an MTL4 buffer.
    ev = MTL.MTLSharedEvent(dev)
    ccall(:objc_msgSend, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, UInt64),
          q4, sel("signalEvent:value:"), rawptr(ev), UInt64(1))
    while ev.signaledValue < 1
        yield()
    end
    GC.@preserve kernel argbuf icb nothing
    return nothing
end

function mtl4_add!(a, v::Float32)
    i = Metal.thread_position_in_grid_1d()
    @inbounds a[i] += v
    return nothing
end

@testset "Metal 4" begin
    dev = Metal.device()
    if !supports(dev, "supportsMTL4CommandQueue")
        @test_skip "this device has no Metal 4 command queue"
    else
        @testset "the model a migration would have to adopt" begin
            for (proto, wanted) in MTL4_SELS
                sels = protocol_selectors(proto)
                @test sels !== nothing
                for w in wanted
                    @test w in sels
                end
            end
            enc = protocol_selectors("MTL4ComputeCommandEncoder")
            # The three absences that ARE the migration: arguments by address,
            # residency by set, hazards by hand.
            @test !any(s -> startswith(s, "setBytes"), enc)
            @test !any(s -> startswith(s, "setBuffer"), enc)
            @test !any(s -> startswith(s, "useResource"), enc)
            # …and a recorded plan is already in that shape: both execute forms are here.
            @test "executeCommandsInBuffer:withRange:" in enc
            @test "executeCommandsInBuffer:indirectBuffer:" in enc
        end

        @testset "a kernel runs through an MTL4 argument table" begin
            a = Metal.MtlVector{Float32}(undef, 64)
            fill!(a, 0f0); Metal.synchronize()
            mtl4_probe!(a, 2.5f0, false)
            @test all(==(2.5f0), Array(a))
        end

        @testset "an indirect command buffer replays on an MTL4 queue" begin
            a = Metal.MtlVector{Float32}(undef, 64)
            fill!(a, 0f0); Metal.synchronize()
            mtl4_probe!(a, 1.5f0, true)
            @test all(==(1.5f0), Array(a))
        end
    end
end
