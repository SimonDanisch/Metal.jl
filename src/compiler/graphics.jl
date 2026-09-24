# Julia → AIR vertex and fragment programs.
#
# Metal.jl compiles Julia to compute kernels only: GPUCompiler's Metal target
# emits a `void` entry with buffer arguments and registers it under the
# `air.kernel` named metadata, and `library.jl` writes `PROGRAM_KERNEL` for
# every function it packs. Neither the compiler nor the metallib writer has ever
# produced a vertex or fragment program.
#
# ── The specification, read out of a shipping metallib ───────────────────────
#
# Not guessed. `/System/Library/Frameworks/PencilKit.framework/.../default.metallib`
# holds 25 functions, 7 of them `air.vertex` and 11 `air.fragment`; the shapes
# below are what our own `MetalLib` reader plus LLVM.jl found in
# `sixChannelBlend_vertex` / `sixChannelBlend_fragment`. Reproduce with
# `read(io, Metal.MetalLib)` and `parse(LLVM.Module, fn.air_module)`.
#
# ENTRY SIGNATURE. Both stages RETURN a packed literal struct — this is the part
# that has no counterpart in the kernel path, where the entry is always `void`:
#
#   vertex   <{ <4 x float>, <2 x float>, <2 x float>, <2 x float> }>
#              (ptr addrspace(1), ptr addrspace(1), ptr addrspace(2), i32)
#   fragment <{ <4 x half>, <4 x half> }>
#              (<4 x float>, <2 x float>, …, ptr addrspace(2), …)
#
# The struct's fields correspond 1:1, in order, to the entries of the OUTPUT
# metadata list.
#
# NAMED METADATA. `air.vertex` / `air.fragment`, each node a triple with exactly
# the shape `air.kernel` already uses — `{ptr @entry, outputs, inputs}` — except
# that for a kernel the middle operand is an EMPTY node (GPUCompiler builds it
# as `stage_infos`, see its `metal.jl`), and here it carries the outputs.
#
# VERTEX OUTPUTS
#   {air.position,      air.arg_type_name, "float4", air.arg_name, "position"}
#   {air.vertex_output, "generated(<mangled>)",
#                       air.arg_type_name, "float2", air.arg_name, "texCoord"}
#
# FRAGMENT OUTPUTS
#   {air.render_target, i32 <index>, air.arg_type_name, "half4",
#                       air.arg_name, "color"}
#
# VERTEX INPUTS — buffers are IDENTICAL to what `add_argument_metadata!` already
# emits for kernels, so that machinery is reusable verbatim:
#   {i32 <n>, air.buffer, air.location_index, i32 <loc>, i32 1, air.read,
#    air.address_space, i32 <as>, air.arg_type_size, i32 <sz>,
#    air.arg_type_align_size, i32 <al>, air.arg_type_name, "<T>",
#    air.arg_name, "<name>"}
#   {i32 <n>, air.vertex_id, air.arg_type_name, "uint", air.arg_name, "vid"}
#
# FRAGMENT INPUTS
#   {i32 0, air.position, air.center, air.no_perspective, air.arg_type_name,
#    "float4", air.arg_name, "position"}
#   {i32 <n>, air.fragment_input, "generated(<mangled>)", air.center,
#    air.perspective, air.arg_type_name, "float2", air.arg_name, "texCoord"}
#   textures: {i32 <n>, air.texture, air.location_index, i32 <loc>, i32 1,
#              air.sample, air.arg_type_name, "texture2d<half, sample>", …}
#
# ── The mesh stage, read out of a shipping metallib the same way ────────────
#
# NOT IMPLEMENTED HERE YET. This is the specification for it, recorded where the
# vertex and fragment one is, and obtained the same way: a scan of the 296
# metallibs under /System/Library found exactly one function with a program type
# this reader has no name for, `particle_gaussian_mesh` in
# `VFX.framework/Versions/A/Resources/default.metallib`. Reproduce with
# `read(io, Metal.MetalLib)` and `parse(LLVM.Module, fn.air_module)`.
#
# PROGRAM TYPE is 7. `ProgramType` in `library.jl` stops at
# `PROGRAM_INTERSECTION = 6`, so this value has no name there yet.
#
# NAMED METADATA is `air.mesh`, and its node has the same three operands as
# `air.kernel` — {ptr @entry, outputs, inputs} — with the OUTPUTS EMPTY. That is
# the structural difference from the vertex and fragment stages above: they
# return a packed struct whose fields match the output list, and a mesh stage
# returns nothing at all. Everything it produces goes through the object it is
# handed.
#
# ENTRY SIGNATURE is `void`, and its FIRST parameter is the output object:
#
#   void (ptr addrspace(7), ptr addrspace(2), …, i32, i32, i32, i16)
#
# Address space 7 is the mesh object's. It is a parameter and not a global,
# which is why an intrinsic that took only a slot could never be lowered here.
#
# THE OBJECT'S ARGUMENT ENTRY carries the bounds as compile-time constants:
#
#   {i32 0, air.mesh,
#    !{"air.mesh_type_info", <vertex type>, <primitive type>,
#      i32 96, i32 32, "air.triangle"},
#    air.arg_type_name, "mesh<particle_vertex_io, particle_primitive_io, 96, 32, triangle>",
#    air.arg_name, "output"}
#
# The two integers are max_vertices and max_primitives, and the string is the
# topology. They are part of the TYPE, which is why `KernelInterface.MeshConfig`
# holds them rather than a runtime: a backend cannot emit this entry without
# them.
#
# THE INTRINSICS, all taking the object as their first argument:
#
#   air.set_position_mesh          (ptr addrspace(7), i32 slot, <4 x float>)
#   air.set_vertex_data_mesh.<T>   (ptr addrspace(7), i32 field, i32 slot, <T>)
#   air.set_primitive_data_mesh.<T>(ptr addrspace(7), i32 field, i32 slot, <T>)
#   air.set_index_mesh             (ptr addrspace(7), i32 slot, i8)
#   air.set_primitive_count_mesh   (ptr addrspace(7), i32)
#
# FIELD BEFORE SLOT in the two data intrinsics, and the other way round in
# `set_position_mesh`. That is measured, not read: see `compiler/mesh.jl` for the
# experiment and why the two orders were indistinguishable until a shader wrote
# more than one vertex.
#   air.set_clip_distance_mesh, air.set_render_target_array_index_mesh.i8,
#   air.set_viewport_array_index_mesh.i8
#
# `<T>` is a suffix per value type — .f16 .i16 .i32 .v2f32 .v2i16 .v3f32 .v3f16
# .v4f16 were present in this one module, so the set is open and driven by what
# the shader writes.
#
# Three things follow for the lowering:
#
#   * POSITION IS ITS OWN INTRINSIC, not field zero of the vertex data. So
#     `set_mesh_vertex!(out, slot, nt)` becomes one `air.set_position_mesh` plus
#     one `air.set_vertex_data_mesh.<T>` per remaining field of the NamedTuple,
#     indexed by field position.
#   * INDICES ARE WRITTEN ONE AT A TIME AND ARE i8. One at a time is why
#     `set_mesh_triangle!` takes the triple and a backend spends three calls:
#     SPIR-V writes a `uvec3` in one, so the triple is the shape that fits both.
#     `i8` is the harder consequence — a threadgroup's output holds at most 256
#     vertices, which `MeshConfig` now refuses to exceed.
#   * PER-PRIMITIVE DATA IS A SEPARATE INTRINSIC from per-vertex data. That is
#     the flat varying, and `KernelInterface`'s vocabulary has no way to write
#     one yet: `emit!(gs, vertex)` reaches `set_vertex_data` only. RayMakie's
#     line shader writes ten of them, so this is the next gap to close, and it
#     needs `varyings` to say which of its fields are flat.
#
# BUILTINS this one used: air.thread_position_in_grid,
# air.thread_index_in_threadgroup, air.threads_per_threadgroup,
# air.amplification_id.
#
# NO OBJECT STAGE was found. No function in any of the 296 libraries had a
# program type of 8 or above, and this mesh shader has none — it reads
# `air.thread_position_in_grid` and is dispatched by the host. So the object
# stage's payload has no ground truth here, which is why `ObjectConfig` declares
# none rather than guessing one.

# ── The one non-obvious thing: how the stages are wired together ─────────────
#
# A varying is matched between stages by the STRING in the second slot, and by
# nothing else — not by position, not by name, not by type. It reads
# `generated(16snapshotTexCoordDv2_f)`: an Itanium-style mangling of
# `<length><name><type>`, so `texCoord` of type `float2` becomes
# `generated(8texCoordDv2_f)`. Vertex output and fragment input must produce
# byte-identical strings or the two stages simply do not connect, and nothing
# reports it as an error.
#
# `mangle_varying` below is that rule, and `test/graphics.jl` pins it against
# the four strings observed in the reference metallib.

"""
    mangle_varying(name::AbstractString, T::Type) -> String

The `generated(...)` tag that links a vertex output to a fragment input.

Itanium-ish: the name with a length prefix, then the type encoding. This is the
ONLY thing that pairs the two stages, so vertex and fragment must derive it from
the same `(name, type)` pair or the varying silently carries garbage.
"""
function mangle_varying(name::AbstractString, T::Type)
    return "generated($(length(name))$(name)$(air_type_mangling(T)))"
end

"""
    air_type_mangling(T) -> String

Itanium mangling for the types a varying can have.

`f`/`h`/`i`/`j` are float/half/int/uint; `Dv<N>_` is the vector prefix, so a
`float2` is `Dv2_f` — which is what the reference metallib carries for its
`float2` varyings. Scalars have no prefix at all.
"""
air_type_mangling(::Type{Float32}) = "f"
air_type_mangling(::Type{Float16}) = "Dh"
air_type_mangling(::Type{Int32})   = "i"
air_type_mangling(::Type{UInt32})  = "j"
# `h` is Itanium's `unsigned char`, which is what a `device uchar*` points at —
# how a VISIBLE function takes an opaque payload buffer. (`Dh` above is `half`;
# the two are distinct manglings that both start with an h.)
air_type_mangling(::Type{UInt8})   = "h"
air_type_mangling(::Type{Int8})    = "a"
air_type_mangling(::Type{NTuple{N,T}}) where {N,T} = "Dv$(N)_" * air_type_mangling(T)

"""
    air_wrapped_vector(T) -> Type or nothing

The `NTuple` inside a one-field vector wrapper, or `nothing` if `T` is not one.

`NTuple` is how a Julia shader spells a vector, but a caller writing a real
shader writes `Vec4f`, which is an `SVector{4,Float32}`, which is a struct whose
single field is an `NTuple{4,Float32}`. So the wrapper is UNWRAPPED rather than
tabulated: naming `SVector` here would make StaticArrays a dependency of the
compiler to recognise four types, and the same test already serves `Point3f`,
`RGBA` and whatever else a caller reaches for.

One field, and that field a homogeneous tuple, is the whole test. Anything else
is still an error, so a struct that merely happens to be small does not silently
become a vector.
"""
function air_wrapped_vector(@nospecialize(T::Type))
    (isconcretetype(T) && fieldcount(T) == 1) || return nothing
    F = fieldtype(T, 1)
    return F <: Tuple && isconcretetype(F) && F === NTuple{fieldcount(F),eltype(F)} ?
           F : nothing
end

function air_type_mangling(@nospecialize(T::Type))
    F = air_wrapped_vector(T)
    F === nothing &&
        error("no AIR type mangling for $T — a varying must be a 32/16-bit scalar, " *
              "a vector of one, or a one-field wrapper around such a vector; " *
              "extend `air_type_mangling` if Metal grows another.")
    return air_type_mangling(F)
end

"""
    air_stage_name(T) -> String

The `air.arg_type_name` string for a stage input or output, e.g. `"float4"`.

Separate from the mangling because the two use different spellings for the same
type: the linkage tag says `Dv2_f`, the human-readable metadata says `float2`,
and the reference metallib carries both for every varying.
"""
air_stage_name(::Type{Float32}) = "float"
air_stage_name(::Type{Float16}) = "half"
air_stage_name(::Type{Int32})   = "int"
air_stage_name(::Type{UInt32})  = "uint"
air_stage_name(::Type{UInt8})   = "uchar"
air_stage_name(::Type{Int8})    = "char"
air_stage_name(::Type{NTuple{N,T}}) where {N,T} = air_stage_name(T) * string(N)
function air_stage_name(@nospecialize(T::Type))
    F = air_wrapped_vector(T)
    F === nothing && error("no AIR type name for $T — see `air_type_mangling`")
    return air_stage_name(F)
end

# ── Rewriting a compiled kernel into a graphics stage ────────────────────────
#
# A stage is compiled as a KERNEL first, and only then reshaped. That is not a
# shortcut, it is the only order that works: every Metal-specific transform in
# GPUCompiler's `finish_ir!` — the parameter and global address-space rewrites,
# `split_aggregate_loads!`, `add_argument_metadata!` — is gated on
# `job.config.kernel`, and a vertex program needs all of them exactly as much as
# a compute one does. Metal.jl's own `finish_ir!` wraps GPUCompiler's through
# `invoke`, so this runs after that work is done and can reshape the result.
#
# The one thing a kernel cannot be is non-void, so the Julia side writes its
# stage output through a trailing pointer argument, kernel-style, and this pass
# turns that argument into the return value.
#
# It does NOT pattern-match the store. Guessing which instruction produces the
# output would break the moment the body writes it field by field, or in a
# branch, or twice. Instead the trailing parameter becomes an `alloca` that the
# cloned body still stores into, and every `ret void` becomes a load from that
# slot followed by a real `ret`. SROA in the clean-up promotes the slot away, so
# the emitted AIR looks as if the value had been returned all along.
#
# The clone idiom (conversion block, `value_map`, `clone_into!`, branch to the
# old entry, erase and rename) is GPUCompiler's own, from
# `add_parameter_address_spaces!`; deviating from it here would only find new
# ways to leave dangling metadata behind.

"""
    stage_output_type(job) -> Type

What the stage returns, read off the Julia signature's trailing pointer argument.

Derived rather than carried in `MetalCompilerParams`, because the two must agree
and a second source of truth is a second thing to get out of sync: the argument
the body writes through IS the declaration of what the stage outputs.
"""
function stage_output_type(@nospecialize(job::CompilerJob))
    sig = job.source.specTypes
    args = sig.parameters[2:end]          # [1] is the function itself
    isempty(args) &&
        error("a graphics stage needs a trailing output pointer argument; the signature has none")
    P = last(args)
    if P <: Core.LLVMPtr
        return P.parameters[1]
    elseif P <: Ptr
        return P.parameters[1]
    end
    error("the last argument of a graphics stage must be a pointer to its output " *
          "struct, got $P")
end

"""
    stage_return!(job, mod, f) -> LLVM.Function

Turn `f(args..., out::Ptr{T})` into `f(args...) -> T`.

Returns the new entry. `f` is erased and the new function takes its name, so
callers must re-look-up the entry afterwards.
"""
function stage_return!(@nospecialize(job::CompilerJob), mod::LLVM.Module, f::LLVM.Function)
    ft = function_type(f)
    params = collect(LLVM.parameters(ft))
    isempty(params) &&
        error("a graphics stage needs a trailing output pointer argument; `$(LLVM.name(f))` has none")
    outparam = last(params)
    outparam isa LLVM.PointerType ||
        error("the last argument of a graphics stage must be the output pointer, got $outparam")

    # What the stage returns, in AIR's spelling. NOT `convert(LLVMType, T)`: an
    # `NTuple{4,Float32}` lowers to `[4 x float]`, and AIR wants `<4 x float>`.
    # The reference metallib's vertex entry returns
    # `<{ <4 x float>, <2 x float>, … }>` — a PACKED struct of VECTORS — and a
    # struct of arrays is a different type that the Metal loader rejects.
    T_jl  = stage_output_type(job)
    # A VISIBLE function returns its one value BARE. Every stage above returns a
    # packed struct because it has several outputs to name; a visible function
    # has exactly one and its caller is MSL, where `float4 f(...)` returns a
    # `<4 x float>` and nothing else. Handing back `<{ <4 x float> }>` links,
    # builds a pipeline, dispatches and completes -- and the caller reads zeros,
    # because a packed one-field struct is a different ABI and nothing checks it.
    T_out = isvisiblestage(job.config.params.stage) ?
            air_field_type(fieldtype(T_jl, 1)) : air_output_struct(T_jl)

    new_ft = LLVM.FunctionType(T_out, params[1:(end - 1)])
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(LLVM.parameters(f), LLVM.parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    slot = nothing
    @dispose builder = IRBuilder() begin
        entry = BasicBlock(new_f, "conversion")
        position!(builder, entry)

        # The body keeps writing to a pointer; it just points at a stack slot now.
        # Cast into the address space the old parameter had, so nothing in the
        # cloned body sees a different pointer type than it was compiled against.
        # The slot keeps JULIA's layout, because that is what the cloned body
        # stores into; the conversion to AIR's vector form happens at the `ret`.
        slot = alloca!(builder, convert(LLVMType, T_jl), "stage_out")
        as = addrspace(outparam)
        outptr = as == 0 ? slot : addrspacecast!(builder, slot, outparam)

        new_args = LLVM.Value[LLVM.parameters(new_f)[i] for i in 1:(length(params) - 1)]
        for i in 1:(length(params) - 1), attr in collect(parameter_attributes(f, i))
            push!(parameter_attributes(new_f, i), attr)
        end
        push!(new_args, outptr)

        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i, param) in enumerate(LLVM.parameters(f))
        )
        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes = LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

        br!(builder, blocks(new_f)[2])
    end

    # Every exit now returns the slot's contents.
    for bb in blocks(new_f), inst in collect(instructions(bb))
        inst isa LLVM.RetInst || continue
        isempty(collect(operands(inst))) || continue      # already returns something
        @dispose builder = IRBuilder() begin
            position!(builder, inst)
            jl = load!(builder, convert(LLVMType, T_jl), slot)
            ret!(builder, T_out isa LLVM.StructType ?
                          to_air_output(builder, jl, T_jl, T_out) :
                          to_air_field(builder, extract_value!(builder, jl, 0),
                                       fieldtype(T_jl, 1)))
        end
        erase!(inst)
    end

    fn = LLVM.name(f)
    GPUCompiler.prune_constexpr_uses!(f)
    @assert isempty(uses(f)) "graphics entry $(fn) still has uses after cloning"
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)
    return new_f
end

"""
    stage_values!(job, mod, f) -> LLVM.Function

Pass a visible function's parameters BY VALUE.

Every other stage takes its arguments the way a kernel does — one
`ptr addrspace(1)` per argument, loaded in the entry block — because that is what
the kernel ABI makes of them and what the binder on the host fills in. A VISIBLE
function has no binder: it is called from another shader, so its parameters are
values in registers and its signature is what the caller links against. Left as
pointers it compiles and tags correctly and then cannot be called at all.

The rewrite is the same clone-and-remap `stage_return!` does for the output
pointer, run the other way: the new entry takes values, and each one is stored
into a stack slot that the cloned body loads from exactly as before. The slot and
its thread-to-device cast are what `stage_cleanup!` promotes away — this is the
house pattern, not a leak, and the store/load pair does not survive it.

A parameter is only converted when its pointee type is unambiguous: every use in
the body is a `load` of one type. Anything else — an aggregate read through a
GEP, a pointer passed along — stays a pointer, because a visible function may
legitimately take one and guessing would silently change the ABI.
"""
function stage_values!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                       f::LLVM.Function)
    params = collect(LLVM.parameters(function_type(f)))
    args   = collect(LLVM.parameters(f))

    # What each parameter is worth as a value, or `nothing` to leave it alone.
    #
    # From the DECLARED Julia type, not from the load in the body. Those differ
    # whenever the body reinterprets: a `Float32` parameter whose only use is
    # `reinterpret(UInt32, x)` folds to a `load i32`, and taking that would give
    # the function an `i32` parameter where its caller — MSL, declaring the table
    # signature by hand — writes `float`. Bit-identical and still wrong: they are
    # different register classes, and nothing checks the two spellings agree.
    jltys = visible_arg_types(job)
    pointee = Vector{Union{Nothing,LLVMType}}(nothing, length(args))
    for (i, arg) in enumerate(args)
        params[i] isa LLVM.PointerType || continue
        if i <= length(jltys)
            # A `Core.LLVMPtr` really is a pointer; leave it one.
            jltys[i] <: Core.LLVMPtr && continue
            # Everything else follows the DECLARATION, whatever the body does
            # with it — including nothing. An unused parameter has no loads to
            # infer from, and leaving it a pointer gives the function a
            # signature its caller does not have: the metadata still says
            # `float` while the entry takes `ptr`, and the call reads garbage or
            # does not happen at all. Neither is reported.
            isbitstype(jltys[i]) || continue
            pointee[i] = convert(LLVMType, jltys[i])
            continue
        end
        # Past the declared arguments: an appended builtin, whose type is
        # whatever it is loaded as.
        us = [user(u) for u in uses(arg)]
        isempty(us) && continue
        all(u -> u isa LLVM.LoadInst, us) || continue
        ts = unique(LLVMType[value_type(u) for u in us])
        length(ts) == 1 || continue
        pointee[i] = only(ts)
    end
    all(isnothing, pointee) && return f

    new_ft = LLVM.FunctionType(LLVM.return_type(function_type(f)),
                               LLVMType[pointee[i] === nothing ? params[i] : pointee[i]
                                        for i in eachindex(params)])
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(args, LLVM.parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    @dispose builder = IRBuilder() begin
        position!(builder, BasicBlock(new_f, "byvalue"))
        new_args = LLVM.Value[]
        for (i, ty) in enumerate(pointee)
            np = LLVM.parameters(new_f)[i]
            if ty === nothing
                # Carried through untouched, attributes and all.
                for attr in collect(parameter_attributes(f, i))
                    push!(parameter_attributes(new_f, i), attr)
                end
                push!(new_args, np)
                continue
            end
            # The slot carries the DECLARED type, and the cloned body loads
            # whatever it was compiled to load — the same 32 bits under a
            # different name when the body reinterprets. `stage_cleanup!`
            # promotes the slot away and the bitcast with it.
            slot = alloca!(builder, ty, LLVM.name(args[i]))
            store!(builder, np, slot)
            as = addrspace(params[i])
            push!(new_args, as == 0 ? slot : addrspacecast!(builder, slot, params[i]))
        end

        value_map = Dict{LLVM.Value, LLVM.Value}(
            param => new_args[i] for (i, param) in enumerate(args))
        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes = LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)
        br!(builder, blocks(new_f)[2])
    end

    fn = LLVM.name(f)
    GPUCompiler.prune_constexpr_uses!(f)
    @assert isempty(uses(f)) "visible function $(fn) still has uses after cloning"
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)
    return new_f
end

# ── The stage metadata ───────────────────────────────────────────────────────
#
# GPUCompiler already produced a well-formed `air.kernel` node for this entry:
# `{ptr @entry, <empty stage_infos>, <arg_infos>}`. A stage node has exactly the
# same three-operand shape, and its third operand — the argument metadata — is
# byte-for-byte what a kernel's is, because a buffer argument is described the
# same way whichever stage reads it (see the specification at the top of this
# file, and GPUCompiler's `add_argument_metadata!`).
#
# So this does not rebuild the node. It takes GPUCompiler's, keeps the argument
# list, drops the entry for the output pointer that `stage_return!` removed,
# fills in the outputs that a kernel leaves empty, and re-registers the whole
# thing under `air.vertex` or `air.fragment`. Rebuilding the argument metadata
# from scratch would mean re-deriving sizes, alignments and address spaces that
# GPUCompiler has already computed correctly.

"""
    isvisiblestage(stage) -> Bool

Whether `stage` compiles to an AIR VISIBLE function — one a kernel calls, rather
than one the rasteriser runs.

Two of them. `:visible` is the plain form: its signature is exactly what the
Julia function declares. `:candidate` is a PROCEDURAL RAY CANDIDATE, which takes
the seven candidate builtins on top, always and in a fixed order, because the MSL
traversal that calls it declares the table's type by hand and cannot know what a
particular payload happens to read. Making that the rule for every visible
function instead was tried and is wrong: it silently re-shaped the signature of
ones that take their arguments explicitly.
"""
isvisiblestage(stage::Symbol) = stage === :visible || stage === :candidate

"""Named metadata a stage is registered under."""
stage_metadata_key(stage::Symbol) =
    stage === :vertex ? "air.vertex" :
    stage === :fragment ? "air.fragment" :
    stage === :mesh ? "air.mesh" :
    isvisiblestage(stage) ? "air.visible" :
    error("no AIR metadata key for stage :$stage")

"""
    stage_outputs(stage, T) -> Vector{Metadata}

The output list: one node per field of the returned struct `T`.

For a vertex the first field is the clip position (`air.position`) and the rest
are varyings tagged with the `generated(...)` linkage string; for a fragment
every field is a render target. Field ORDER is the contract — the metadata list
and the struct's fields are matched positionally, not by name.
"""
function stage_outputs(stage::Symbol, @nospecialize(T::Type))
    # A mesh stage returns nothing at all: everything it produces goes through
    # the object it was handed, so its `air.mesh` node carries an EMPTY output
    # list. That is the structural difference from the two stages below, and it
    # is what the shipping metallib shows.
    stage === :mesh && return Metadata[]
    # A VISIBLE function is not a stage at all: it is an ordinary AIR function a
    # kernel calls, so it has ONE return value and no per-target or per-varying
    # tagging. `air.visible_output` carries just the type — no index, no name,
    # which is the shape `CC_InlineCompositing32x32` shows in CoreComposite's
    # shipped `default-cc.metallib`.
    #
    # The single field is UNWRAPPED here. Every other stage returns a packed
    # struct because it has several outputs; a visible function has one, and a
    # caller declaring `extern float4 f(...)` links against the bare type.
    if isvisiblestage(stage)
        names = fieldnames(T)
        length(names) == 1 ||
            error("a visible function returns ONE value; got $(length(names)): $names")
        return Metadata[MDNode(Metadata[MDString("air.visible_output"),
                                        MDString("air.arg_type_name"),
                                        MDString(air_stage_name(fieldtype(T, 1)))])]
    end
    names = fieldnames(T)
    isempty(names) && error("a graphics stage must return at least one value")
    out = Metadata[]
    for (i, name) in enumerate(names)
        F = fieldtype(T, i)
        md = Metadata[]
        if stage === :vertex && i == 1
            push!(md, MDString("air.position"))
        elseif stage === :vertex
            push!(md, MDString("air.vertex_output"))
            push!(md, MDString(mangle_varying(String(name), F)))
        else
            # TWO integers, not one: `{air.render_target, i32 <index>, i32 0, …}`.
            # Read off a literal-index fragment in the wild —
            # `path_exterior_fragment` carries
            # `!{!"air.render_target", i32 1, i32 0, !"air.arg_type_name", …}`.
            # PencilKit's has a function-constant pointer where the first
            # integer is, which is what hid the second one.
            #
            # Emitting only the index does not fail to build: the metallib loads
            # and reports `MTLFunctionTypeFragment`, and then creating a render
            # pipeline from it dies with "Internal compiler error
            # (AGXMetalG17G, code 3)" and nothing that names the cause.
            push!(md, MDString("air.render_target"))
            push!(md, Metadata(ConstantInt(Int32(i - 1))))
            push!(md, Metadata(ConstantInt(Int32(0))))
        end
        push!(md, MDString("air.arg_type_name"))
        push!(md, MDString(air_stage_name(F)))
        push!(md, MDString("air.arg_name"))
        push!(md, MDString(String(name)))
        push!(out, MDNode(md))
    end
    return out
end

"""
    shift_buffer_location(node, by) -> MDNode

`node` with its `air.location_index` moved by `by`, or `node` unchanged if it
does not carry one.

Only an `air.buffer`/`air.texture` entry has a location; a builtin like
`air.thread_index_in_threadgroup` is not bound by the host and has none. The
LEADING operand of the node is the parameter position and is deliberately left
alone: a parameter does not move because a binding does.
"""
function shift_buffer_location(node::LLVM.MDNode, by::Integer)
    ops = collect(LLVM.operands(node))
    i = findfirst(o -> o isa LLVM.MDString && string(o) == "air.location_index", ops)
    i === nothing && return node
    i < length(ops) ||
        error("air.location_index is the last operand of an argument node; " *
              "it must be followed by the index itself")
    old = convert(Int, LLVM.Value(ops[i + 1]))
    ops[i + 1] = Metadata(ConstantInt(Int32(old + by)))
    return MDNode(ops)
end

"""
    retag_stage!(job, mod, entry, stage, T_out)

Move `entry` from `air.kernel` to the stage's own named metadata, with outputs.
"""
function retag_stage!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                      entry::LLVM.Function, stage::Symbol, @nospecialize(T_out::Type),
                      markers::Vector{Union{Nothing,Type}} = Union{Nothing,Type}[])
    md = LLVM.metadata(mod)
    haskey(md, "air.kernel") ||
        error("no air.kernel metadata to convert — did GPUCompiler's finish_ir! run?")

    # The node for THIS entry. A module can hold several after deferred codegen.
    kernel_nodes = collect(LLVM.operands(md["air.kernel"]))
    # The operand is the function wrapped as metadata, so the comparison has to
    # be made on that side — `Metadata(entry)` — rather than by unwrapping. By
    # now `replace_metadata_uses!` has already repointed the node at the
    # rewritten function, so this matches the NEW entry.
    want = Metadata(entry)
    idx = findfirst(kernel_nodes) do node
        ops = LLVM.operands(node)
        !isempty(ops) && ops[1] == want
    end
    idx === nothing &&
        error("entry $(LLVM.name(entry)) has no air.kernel node to convert")

    ops = collect(LLVM.operands(kernel_nodes[idx]))
    arg_infos = collect(LLVM.operands(ops[3]))

    # GPUCompiler described the ORIGINAL Julia signature. Two things happened to
    # it since: `stage_return!` dropped the trailing output pointer, and
    # `stage_builtins!` appended one parameter per referenced builtin. The
    # metadata has to follow both, or the entries stop describing the parameters
    # they are indexed against.
    isempty(arg_infos) && error("no argument metadata to adapt")
    # A mesh stage keeps every parameter it started with: nothing was dropped,
    # because there was no output pointer to drop.
    stage === :mesh || pop!(arg_infos)                # the output pointer
    nparams = length(collect(LLVM.parameters(function_type(entry))))
    while length(arg_infos) < nparams
        # A placeholder per appended builtin; `stage_input_metadata!` replaces it
        # with the real tag below, since every one of them is a marked input.
        push!(arg_infos, MDNode(Metadata[]))
    end
    length(arg_infos) == nparams ||
        error("argument metadata does not match the rewritten signature: " *
              "$(length(arg_infos)) entries for $nparams parameters")

    # A visible function's parameters are VALUES, not bindings. GPUCompiler
    # described them as buffers because that is what the kernel ABI makes of
    # every argument; `air.visible_input` is what AIR wants, and the whole list
    # is rebuilt rather than patched because none of the buffer description
    # survives — no location, no address space, just index/type/name.
    #
    # INCOMPLETE, and this is the remaining gap: the metadata says `float`, the
    # SIGNATURE still says `ptr addrspace(1)`, because the kernel ABI passes
    # every argument by pointer and nothing here undoes that. A caller linking
    # `extern float4 f(uint, float3, float3, float)` would pass values where the
    # body loads pointers. What is needed is a by-value rewrite of the entry —
    # the same clone-and-remap shape `stage_builtins!` already does for appended
    # builtins and `stage_return!` for the output pointer, run the other way.
    # Until then this emits a well-formed `PROGRAM_VISIBLE` function with the
    # right output type that cannot yet be CALLED with a value signature.
    if isvisiblestage(stage)
        ptys = collect(LLVM.parameters(function_type(entry)))
        # Named from the JULIA types where there are any, because LLVM cannot
        # tell `uint` from `int` — both are `i32`, and Apple's own visible
        # functions spell the unsigned one `uint`. `air_visible_type_name` is the
        # fallback for a parameter with no Julia counterpart.
        jltys = visible_arg_types(job)
        # The parameters `stage_builtins!` APPENDED have no Julia argument to be
        # named from; their marker carries the AIR spelling instead. Without this
        # a `CandidatePrim` is named `int`, because LLVM has only `i32` to go on,
        # while the MSL caller declares `uint`.
        aligned = stage_align_markers(markers, length(ptys))
        arg_infos = Metadata[
            MDNode(Metadata[Metadata(ConstantInt(Int32(i - 1))),
                            MDString("air.visible_input"),
                            MDString("air.arg_type_name"),
                            MDString(i <= length(jltys) ?
                                     air_visible_arg_name(jltys[i], ptys[i]) :
                                     (aligned[i] === nothing ?
                                      air_visible_type_name(ptys[i]) :
                                      last(stage_input_tag(aligned[i])))),
                            MDString("air.arg_name"),
                            MDString(string("arg", i - 1))])
            for i in eachindex(ptys)]
    end

    isempty(markers) || isvisiblestage(stage) ||
        stage_input_metadata!(arg_infos, stage_align_markers(markers, length(arg_infos)))

    # A texture and a sampler are both `ptr addrspace(1)`/`ptr addrspace(2)` to the
    # kernel ABI, so GPUCompiler described them as buffers. Left that way Metal
    # binds a buffer where the stage expects a texture, and the pipeline fails to
    # build with an internal compiler error naming nothing.
    let (kinds, argtypes) = texture_binding_parameters(job)
        any(!=(0), kinds) &&
            texture_argument_metadata!(arg_infos, kinds, argtypes, length(markers))
    end

    # The object a mesh stage writes through. It is the FIRST parameter, and
    # GPUCompiler described it as a buffer, which is what the kernel ABI made it;
    # AIR needs `air.mesh` plus the whole type of the object, and that type is
    # what bounds the stage.
    #
    # Position 1 and not an offset from the end. Computing it from the end
    # overwrote whichever builtin happened to land there — with the compute
    # builtins threaded in, that was `air.thread_index_in_threadgroup`, so every
    # thread in the group read index 0, wrote to the same slots, and the stage
    # behaved as if it ran once. The object stayed described as a buffer at
    # location 0 on top of that, which is what made a stage buffer bound at Metal
    # slot 0 collide with it. Neither is a rule of Apple's: MSL puts a mesh
    # stage's own buffer at `buffer(0)` and it works.
    if stage === :mesh
        arg_infos[1] = air_mesh_argument(mesh_object_type(job))
        # …and the buffers behind it move down one slot, because the object is
        # not a buffer and does not consume a binding. So a mesh stage's buffers
        # are numbered from zero exactly like a vertex stage's, and nothing above
        # this backend has to know a mesh stage is different.
        for i in 2:length(arg_infos)
            arg_infos[i] = shift_buffer_location(arg_infos[i], -1)
        end
    end

    node = MDNode(Metadata[Metadata(entry),
                           MDNode(stage_outputs(stage, T_out)),
                           MDNode(arg_infos)])
    push!(md[stage_metadata_key(stage)], node)

    # …and it must stop being a kernel. Leaving the old node behind would leave
    # `air.kernel` pointing at a function that now returns a struct, which is not
    # a kernel by any reading. Named metadata has no per-operand removal, so the
    # list is emptied and the OTHER entries — deferred codegen can leave several
    # in one module — are pushed back.
    kept = [n for (i, n) in enumerate(kernel_nodes) if i != idx]
    empty!(md["air.kernel"])
    for n in kept
        push!(md["air.kernel"], n)
    end
    return nothing
end

"""
    air_output_struct(T) -> LLVM.StructType

The AIR type a stage returns: a PACKED struct whose fields are LLVM vectors.

`convert(LLVMType, ::NTuple{4,Float32})` gives `[4 x float]`, an ARRAY. AIR
spells the same thing `<4 x float>`, and the reference metallib's entries return
`<{ <4 x float>, <2 x float>, … }>`. The two are distinct LLVM types and the
loader only accepts the second.
"""
function air_output_struct(@nospecialize(T::Type))
    fields = LLVMType[]
    for i in 1:fieldcount(T)
        push!(fields, air_field_type(fieldtype(T, i)))
    end
    # A single output is still WRAPPED here, though every shipped single-target
    # fragment returns the vector bare — `TextureCopy` returns `<4 x float>`, and
    # the packed struct appears only with several fields. The one-field struct is
    # what this backend has always emitted and what every test draws through, so
    # it is left alone: returning it bare was tried while chasing the texture
    # crash in `compiler/texture.jl` and changed nothing.
    return LLVM.StructType(fields; packed = true)
end

"""
    visible_arg_types(job) -> Vector{Union{Nothing,Type}}

The Julia types of a visible function's parameters, trailing output pointer
dropped — the same slice `stage_input_types` takes, kept whole rather than
reduced to stage markers, because here every one of them names a type in AIR.
"""
function visible_arg_types(@nospecialize(job::CompilerJob))
    args = collect(job.source.specTypes.parameters[2:end])
    isempty(args) || pop!(args)                 # the output pointer
    return Union{Nothing,Type}[a for a in args]
end

"""
    air_visible_arg_name(jl, llvm) -> String

What AIR calls one visible-function parameter.

From the JULIA type where there is one, because LLVM cannot tell `uint` from
`int` — both are `i32` — and Apple's own visible functions spell the unsigned one
`uint`. A `Core.LLVMPtr{T,AS}` is a DEVICE POINTER and has no `air_stage_name`;
it is named for what it points at, which is how MSL spells the parameter the
caller must declare (`device float*`).
"""
air_visible_arg_name(::Nothing, llvm::LLVMType) = air_visible_type_name(llvm)
air_visible_arg_name(@nospecialize(jl::Type), llvm::LLVMType) =
    jl <: Core.LLVMPtr ? air_stage_name(first(jl.parameters)) * "*" :
    air_stage_name(jl)

"""
    air_visible_type_name(t::LLVMType) -> String

What AIR calls one visible-function parameter, read off the LLVM type.

Off the LLVM type and not the Julia one, because by this point the signature has
already been rewritten — this is the type the CALLER will link against, and a
mismatch is a link failure with no diagnostic rather than a wrong answer.
"""
function air_visible_type_name(t::LLVMType)
    t isa LLVM.PointerType && return "void*"
    if t isa LLVM.VectorType
        base = air_visible_type_name(LLVM.eltype(t))
        return "$base$(Int(length(t)))"
    end
    t == LLVM.FloatType()  && return "float"
    t == LLVM.HalfType()   && return "half"
    t == LLVM.Int32Type()  && return "int"
    t == LLVM.Int16Type()  && return "short"
    t == LLVM.Int8Type()   && return "char"
    t == LLVM.Int1Type()   && return "bool"
    t == LLVM.VoidType()   && return "void"
    error("no AIR visible-function type name for $t")
end

"""AIR's type for one stage output field: a vector for an `NTuple`, else itself."""
air_field_type(::Type{NTuple{N,T}}) where {N,T} =
    LLVM.VectorType(convert(LLVMType, T), N)
function air_field_type(@nospecialize(T::Type))
    F = air_wrapped_vector(T)
    return F === nothing ? convert(LLVMType, T) : air_field_type(F)
end

"""
    to_air_output(builder, val, T_jl, T_air) -> LLVM.Value

Repack a Julia-shaped output value into AIR's vector-of-fields form.

Field by field, and element by element within a field, because there is no
bitcast between `[4 x float]` and `<4 x float>` — LLVM treats an aggregate and a
vector as unrelated. The shuffle is free after `instcombine`; what matters is
that the emitted type is the one AIR names.
"""
function to_air_output(builder::IRBuilder, val::LLVM.Value,
                       @nospecialize(T_jl::Type), T_air::LLVM.StructType)
    out = LLVM.UndefValue(T_air)
    for i in 1:fieldcount(T_jl)
        F = fieldtype(T_jl, i)
        fld = extract_value!(builder, val, i - 1)
        out = insert_value!(builder, out, to_air_field(builder, fld, F), i - 1)
    end
    return out
end

"""One output field: an `NTuple` becomes a vector, anything else passes through."""
function to_air_field(builder::IRBuilder, val::LLVM.Value, ::Type{NTuple{N,T}}) where {N,T}
    vec = LLVM.UndefValue(LLVM.VectorType(convert(LLVMType, T), N))
    for j in 1:N
        elt = extract_value!(builder, val, j - 1)
        vec = insert_element!(builder, vec, elt, ConstantInt(Int32(j - 1)))
    end
    return vec
end
# A wrapper — `Vec4f` and friends — is one `extract_value!` away from the tuple
# the method above handles. LLVM sees `{ [4 x float] }`, so the field has to come
# out before the elements can.
function to_air_field(builder::IRBuilder, val::LLVM.Value, @nospecialize(T::Type))
    F = air_wrapped_vector(T)
    F === nothing && return val
    return to_air_field(builder, extract_value!(builder, val, 0), F)
end

# ── Stage inputs ─────────────────────────────────────────────────────────────
#
# A buffer argument needs no help: GPUCompiler's kernel ABI already passes it as
# `ptr addrspace(1)` with exactly the `air.buffer` metadata AIR wants. A STAGE
# input does — the vertex id, the interpolated fragment position, a varying —
# because those arrive as VALUES in a register, and the kernel ABI passes every
# scalar through a buffer instead. Declaring `vid::UInt32` gets you
# `ptr addrspace(1)`, which is a buffer holding a number, not `[[vertex_id]]`.
#
# The marker types below say which is which. They are ordinary immutable structs
# so a shader reads `vid.value`, and the pass turns the parameter into a bare
# scalar, spilling it to a stack slot so the cloned body's loads still typecheck.
# Same trick as the output, in reverse; SROA removes the slot again.

"""The vertex index, `[[vertex_id]]` in MSL. Read `.value`."""
struct VertexID
    value::UInt32
end

"""The instance index, `[[instance_id]]` in MSL. Read `.value`."""
struct InstanceID
    value::UInt32
end

"""The interpolated clip position a fragment shader receives, `[[position]]`."""
struct FragCoord
    value::NTuple{4,Float32}
end

"""
    Varying{name, T}

One interpolated value a fragment stage reads from the vertex stage. Read
`.value`.

A varying is not a builtin — there is a different one per pipeline — so it is
parameterised by the NAME the vertex stage gave it and by its type, and those
two are exactly what `mangle_varying` needs. The stages link by that mangled
string and by nothing else, so declaring `Varying{:tint, NTuple{4,Float32}}`
here and `tint = NTuple{4,Float32}` in the vertex output is what connects them;
a mismatch in either half silently produces a fragment stage that reads
nothing.

One marker per varying rather than one struct holding all of them, because AIR
delivers each as its OWN entry parameter with its own tag — which is also what
lets the existing one-marker-per-parameter machinery serve this unchanged.
"""
struct Varying{name, T}
    value::T
end

# ── The procedural-candidate builtins ────────────────────────────────────────
#
# What a ray query is OFFERING, inside a visible function called from an MSL
# traversal loop. Ambient for the same reason `vertex_index()` is: the portable
# protocol (`Mantle.candidate_primitive_index` and friends) is a zero-argument
# call, because on Vulkan it reads the inline ray query that is already in
# scope. Here there is no query in scope — it lives in the MSL caller — so the
# values arrive as parameters, and these globals are how a body asks for them
# without spelling them in its signature.

"""The primitive the traversal is offering, ZERO-based as AIR reports it."""
struct CandidatePrim
    value::UInt32
end

# SCALARS, one per component, and not a `float3`. MSL's `float3` is 16 bytes
# with a padding lane; AIR's `<3 x float>` is 12. A visible function table
# declares its signature in MSL and the function is compiled from Julia, so the
# two spellings have to agree at the ABI and nothing checks that they do — the
# call is made, returns UNDEF, and the traversal commits nothing.
"""One component of the candidate's ray origin, in OBJECT space."""
struct CandidateOriginX; value::Float32; end
@doc (@doc CandidateOriginX) struct CandidateOriginY; value::Float32; end
@doc (@doc CandidateOriginX) struct CandidateOriginZ; value::Float32; end

"""One component of the candidate's ray direction, in OBJECT space."""
struct CandidateDirX; value::Float32; end
@doc (@doc CandidateDirX) struct CandidateDirY; value::Float32; end
@doc (@doc CandidateDirX) struct CandidateDirZ; value::Float32; end

const STAGE_INPUTS = Union{VertexID, InstanceID, FragCoord, Varying, CandidatePrim,
                           CandidateOriginX, CandidateOriginY, CandidateOriginZ,
                           CandidateDirX, CandidateDirY, CandidateDirZ}

"""What `air.*` tag a stage-input marker carries, and how AIR names its type."""
stage_input_tag(::Type{VertexID})   = ("air.vertex_id",   "uint")
stage_input_tag(::Type{InstanceID}) = ("air.instance_id", "uint")
stage_input_tag(::Type{FragCoord})  = ("air.position",    "float4")
# A visible function's parameters carry no builtin tag — `retag_stage!` writes
# `air.visible_input` for all of them — so these are named only for the AIR type.
stage_input_tag(::Type{CandidatePrim}) = ("air.visible_input", "uint")
for C in (:CandidateOriginX, :CandidateOriginY, :CandidateOriginZ,
          :CandidateDirX, :CandidateDirY, :CandidateDirZ)
    @eval stage_input_tag(::Type{$C}) = ("air.visible_input", "float")
end
stage_input_tag(::Type{Varying{name,T}}) where {name,T} =
    ("air.fragment_input", air_stage_name(T))

"""The scalar an input marker is passed as."""
stage_input_llvmtype(::Type{VertexID})   = convert(LLVMType, UInt32)
stage_input_llvmtype(::Type{InstanceID}) = convert(LLVMType, UInt32)
stage_input_llvmtype(::Type{FragCoord})  = LLVM.VectorType(convert(LLVMType, Float32), 4)
stage_input_llvmtype(::Type{CandidatePrim}) = convert(LLVMType, UInt32)
for C in (:CandidateOriginX, :CandidateOriginY, :CandidateOriginZ,
          :CandidateDirX, :CandidateDirY, :CandidateDirZ)
    @eval stage_input_llvmtype(::Type{$C}) = convert(LLVMType, Float32)
end
stage_input_llvmtype(::Type{Varying{name,T}}) where {name,T} = air_field_type(T)

"""
    stage_input_types(job) -> Vector{Union{Nothing,Type}}

Per Julia argument: the marker type if it is a stage input, `nothing` otherwise.

The trailing output pointer is excluded — `stage_return!` has already removed it
by the time this matters.
"""
function stage_input_types(@nospecialize(job::CompilerJob))
    args = collect(job.source.specTypes.parameters[2:end])
    # A mesh stage has no trailing output pointer — it writes through the object
    # that is its FIRST argument — so there is nothing to drop here, and that
    # argument is retagged by `air_mesh_argument` rather than marked.
    job.config.params.stage === :mesh || pop!(args)
    return Union{Nothing,Type}[a <: STAGE_INPUTS ? a : nothing for a in args]
end

"""
    stage_align_markers(markers, nparams) -> Vector{Union{Nothing,Type}}

Right-align the per-Julia-argument markers against the LLVM parameter list.

GPUCompiler's kernel ABI prepends a kernel-state pointer that no Julia argument
corresponds to, so parameter `i` is Julia argument `i - offset`. Padding at the
front rather than assuming a fixed offset keeps this correct if the ABI ever
grows or drops an implicit parameter.
"""
function stage_align_markers(markers::Vector{Union{Nothing,Type}}, nparams::Int)
    offset = nparams - length(markers)
    offset >= 0 ||
        error("$(length(markers)) markers for only $nparams parameters")
    return Union{Nothing,Type}[i <= offset ? nothing : markers[i - offset]
                               for i in 1:nparams]
end

"""
    stage_inputs!(job, mod, f, markers) -> LLVM.Function

Turn every parameter named by `markers` from a buffer pointer into a value.

`markers[i]` is the marker type for parameter `i`, or `nothing` to leave it
alone. The value is spilled to a stack slot and the slot handed to the cloned
body, so nothing inside has to know the calling convention changed.
"""
function stage_inputs!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                       f::LLVM.Function, markers::Vector{Union{Nothing,Type}})
    any(!isnothing, markers) || return f

    ft = function_type(f)
    params = collect(LLVM.parameters(ft))
    # GPUCompiler prepends the kernel-state pointer (`kernel_state_to_reference!`),
    # which has no Julia argument, so the markers are right-aligned against the
    # parameter list rather than starting at 1.
    markers = stage_align_markers(markers, length(params))

    # A parameter that ALREADY has the scalar type is one `stage_builtins!`
    # appended — those arrive as values, not as buffer pointers, so there is
    # nothing to convert and marking them again would try to `addrspace` an
    # `i32`. Only the marker-typed Julia arguments need the rewrite.
    needs = [markers[i] !== nothing && params[i] != stage_input_llvmtype(markers[i])
             for i in eachindex(params)]
    any(needs) || return f          # nothing left to convert
    new_types = LLVMType[needs[i] ? stage_input_llvmtype(markers[i]) : params[i]
                         for i in eachindex(params)]
    new_ft = LLVM.FunctionType(LLVM.return_type(ft), new_types)
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(LLVM.parameters(f), LLVM.parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end

    @dispose builder = IRBuilder() begin
        entry = BasicBlock(new_f, "stage_inputs")
        position!(builder, entry)

        new_args = LLVM.Value[]
        for (i, param) in enumerate(params)
            arg = LLVM.parameters(new_f)[i]
            if !needs[i]
                push!(new_args, arg)
                for attr in collect(parameter_attributes(f, i))
                    push!(parameter_attributes(new_f, i), attr)
                end
            else
                # The body was compiled against a pointer to the marker struct,
                # which is a one-field wrapper around the scalar — so a slot
                # holding the scalar has the same layout, and a cast to the old
                # parameter's address space keeps every load in the body valid.
                slot = alloca!(builder, value_type(arg))
                store!(builder, arg, slot)
                as = addrspace(param)
                push!(new_args, as == 0 ? slot : addrspacecast!(builder, slot, param))
            end
        end

        value_map = Dict{LLVM.Value, LLVM.Value}(
            p => new_args[i] for (i, p) in enumerate(LLVM.parameters(f))
        )
        value_map[f] = new_f
        clone_into!(new_f, f; value_map,
                    changes = LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)
        br!(builder, blocks(new_f)[2])
    end

    fn = LLVM.name(f)
    GPUCompiler.prune_constexpr_uses!(f)
    @assert isempty(uses(f)) "stage entry $(fn) still has uses after cloning"
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)
    return new_f
end

"""
    stage_input_metadata!(arg_infos, markers)

Replace the `air.buffer` description of each stage input with its own tag.

GPUCompiler described those arguments as buffers because that is what the kernel
ABI made them; after `stage_inputs!` they are values, and the metadata has to
say so or Metal binds a buffer to a register operand.
"""
function stage_input_metadata!(arg_infos::Vector, markers::Vector{Union{Nothing,Type}})
    for (i, m) in enumerate(markers)
        m === nothing && continue
        tag, typename = stage_input_tag(m)
        md = Metadata[Metadata(ConstantInt(Int32(i - 1))), MDString(tag)]
        argname = string(nameof(m))
        # A fragment's interpolated position also declares how it is sampled;
        # the reference metallib pairs `air.position` with `air.center` and
        # `air.no_perspective`.
        if m === FragCoord
            push!(md, MDString("air.center"))
            push!(md, MDString("air.no_perspective"))
        elseif m <: Varying
            # The linkage string first, then the sampling — the order the
            # reference metallib's fragment inputs carry:
            #   {i32 n, air.fragment_input, "generated(…)", air.center,
            #    air.perspective, air.arg_type_name, "float2", air.arg_name, …}
            name, T = m.parameters
            push!(md, MDString(mangle_varying(string(name), T)))
            push!(md, MDString("air.center"))
            push!(md, MDString("air.perspective"))
            argname = string(name)
        end
        append!(md, Metadata[MDString("air.arg_type_name"), MDString(typename),
                             MDString("air.arg_name"), MDString(argname)])
        arg_infos[i] = MDNode(md)
    end
    return arg_infos
end

"""
    air_program_type(job) -> ProgramType

The metallib function tag for this job's stage.

The library format has always distinguished vertex, fragment and kernel
programs; the writer simply wrote `PROGRAM_KERNEL` for everything, because
nothing else could be produced. A stage packed under the kernel tag is refused
by the loader.
"""
air_program_type(@nospecialize(job::CompilerJob)) = PROGRAM_KERNEL
function air_program_type(@nospecialize(job::MetalCompilerJob))
    stage = job.config.params.stage
    return stage === :vertex   ? PROGRAM_VERTEX :
           stage === :fragment ? PROGRAM_FRAGMENT :
           stage === :mesh     ? PROGRAM_MESH :
           isvisiblestage(stage) ? PROGRAM_VISIBLE :
                                 PROGRAM_KERNEL
end

"""
    stage_cleanup!(job, mod)

Promote the slots the rewrites introduced, and with them the address-space casts.

Both passes hand the cloned body a pointer where it expects one — an `alloca`
for the output, another for each stage input — and cast it into the address
space the parameter had. That cast is `thread` to `device`, which is not a legal
thing to hold on Metal: the store goes to a stack slot the device pointer only
pretends to name.

It is meant to be temporary. SROA promotes the slot into registers and the cast
disappears with it, leaving what the reference shaders contain — a value built
with `insertvalue` and returned, no memory at all:

    %13 = insertvalue <{ <4 x float> }> undef, <4 x float> %12, 0
    ret <{ <4 x float> }> %13

Without this the vertex program still compiles, still links, and still builds a
render pipeline — and draws nothing, because the position was written to a stack
slot through a device pointer. Nothing reports it.

The pass list is GPUCompiler's own from `add_parameter_address_spaces!`, which
introduces the same shape and cleans up after itself the same way.
"""
function stage_cleanup!(@nospecialize(job::CompilerJob), mod::LLVM.Module)
    @dispose pb = NewPMPassBuilder() begin
        add!(pb, NewPMFunctionPassManager()) do fpm
            add!(fpm, SimplifyCFGPass())
            add!(fpm, SROAPass())
            add!(fpm, EarlyCSEPass())
            add!(fpm, InstCombinePass())
        end
        run!(pb, mod)
    end
    return nothing
end

"""
Empty the exception-signalling runtime function.

GPUCompiler ends every throw site it lowers with a call to
`gpu_signal_exception`, unconditionally — the debug level gates the REPORTING
calls before it, not this one. Its body writes the `KernelState` exception
mailbox, and a graphics stage has no kernel state, so the call leaves
`julia.gpu.state_getter` unresolved and validation rejects the module.

There is nothing for it to signal: no mailbox buffer is bound to a render
pipeline and no host call reads one after a draw. So the body goes and the
`llvm.trap` GPUCompiler emits right after it stays, which is what actually stops
an out-of-bounds lane.

Run BEFORE `stage_cleanup!`, so the now-empty function is inlined away rather
than left as a call in the shader's hot path.
"""
function stage_drop_exception_signal!(mod::LLVM.Module)
    haskey(functions(mod), "gpu_signal_exception") || return false
    f = functions(mod)["gpu_signal_exception"]
    isdeclaration(f) && return false
    empty!(f)
    @dispose builder = IRBuilder() begin
        position!(builder, BasicBlock(f, "entry"))
        ret!(builder)
    end
    # It does nothing now, and saying so lets the inliner and DCE treat it as
    # the no-op it is instead of a call that might write memory.
    push!(function_attributes(f), EnumAttribute("alwaysinline", 0))
    # …and PRIVATE, so it is not exported. A stage compiled on its own never
    # noticed: its metallib is linked with nothing. A VISIBLE function is linked
    # INTO a kernel's pipeline, and that kernel carries its own
    # `gpu_signal_exception` — two external definitions of the same name, which
    # the Metal linker refuses with "symbol multiply defined" and no hint that an
    # emptied helper is what collided.
    linkage!(f, LLVM.API.LLVMPrivateLinkage)
    return true
end

# ── Implicit stage inputs ────────────────────────────────────────────────────
#
# `vertex_index()` is a zero-argument call in the shader and a PARAMETER in AIR.
# Something has to bridge that, because AIR has no ambient vertex id — it is an
# entry argument tagged `air.vertex_id` and nothing else.
#
# The shader-side half is `device/intrinsics/graphics.jl`: each builtin loads an
# external global that never gets a definition. This pass is the other half. For
# every such global the module actually references, it appends a parameter of
# the matching type, replaces the loads with it, and reports the marker so
# `retag_stage!` tags the argument.
#
# Same shape as Lava's `addrspace(7)` globals, which its SPIR-V emitter turns
# into BuiltIn Input variables — so a shader written against `vertex_index()`
# and `frag_coord_x()` compiles unchanged on either backend, which is the point.

"""Builtin global → (marker type, LLVM type of the parameter)."""
const STAGE_BUILTINS = Dict(
    "__air_stage_vertex_id"   => VertexID,
    "__air_stage_instance_id" => InstanceID,
    "__air_stage_frag_coord"  => FragCoord,
    # Visible functions only. They carry no `air.*` tag — a visible function's
    # parameters are ordinary values, so these become plain `air.visible_input`
    # entries like every other one.
    # Appended SORTED BY NAME, so the caller's argument order is
    # dx, dy, dz, ox, oy, oz, prim — alphabetical, not declaration order.
    "__air_candidate_dx"   => CandidateDirX,
    "__air_candidate_dy"   => CandidateDirY,
    "__air_candidate_dz"   => CandidateDirZ,
    "__air_candidate_ox"   => CandidateOriginX,
    "__air_candidate_oy"   => CandidateOriginY,
    "__air_candidate_oz"   => CandidateOriginZ,
    "__air_candidate_prim" => CandidatePrim,
)

"""
    stage_builtins!(job, mod, f) -> (LLVM.Function, Vector{Type})

Turn every referenced stage-builtin global into a trailing entry parameter.

Returns the rewritten entry and the marker types, in the order the parameters
were appended, for `stage_input_types` to concatenate.

Only globals that are USED are turned into parameters: an unreferenced builtin
would otherwise cost a register and, worse, an argument slot that shifts every
buffer index after it.
"""
function stage_builtins!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                         f::LLVM.Function)
    used = Tuple{String,Type,LLVM.GlobalVariable}[]
    for (name, marker) in STAGE_BUILTINS
        # A VISIBLE function takes the candidate builtins whether its body reads
        # them or not. Every other stage appends only what it USES, because an
        # unreferenced builtin would cost a register and shift every buffer
        # index after it — there is no caller with a fixed idea of the signature.
        #
        # A visible function has exactly that: the MSL traversal declares the
        # table's type by hand and passes all seven. A payload whose candidate
        # happens to read only `o[1]` would otherwise be compiled to take one
        # float where the caller passes six, and the call returns UNDEF — no
        # error, no diagnostic, and a traversal that commits nothing.
        alwayson = job.config.params.stage === :candidate &&
                   startswith(name, "__air_candidate_")
        if !haskey(globals(mod), name)
            alwayson || continue
            # Declare it so there is something to turn into a parameter. With no
            # uses the rewrite below simply drops the load, which is what an
            # unread builtin should cost.
            GlobalVariable(mod, stage_input_llvmtype(marker), name)
        end
        gv = globals(mod)[name]
        (alwayson || !isempty(uses(gv))) || continue
        push!(used, (name, marker, gv))
    end
    isempty(used) && return (f, Type[])
    # Deterministic order: a `Dict` iterates arbitrarily, and the parameter
    # order IS the argument-metadata order, so it must not vary between runs.
    sort!(used; by = first)

    ft = function_type(f)
    params = collect(LLVM.parameters(ft))
    extra = LLVMType[stage_input_llvmtype(m) for (_, m, _) in used]
    new_ft = LLVM.FunctionType(LLVM.return_type(ft), vcat(params, extra))
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (arg, new_arg) in zip(LLVM.parameters(f), LLVM.parameters(new_f))
        LLVM.name!(new_arg, LLVM.name(arg))
    end
    for (i, (name, _, _)) in enumerate(used)
        LLVM.name!(LLVM.parameters(new_f)[length(params) + i], name)
    end

    value_map = Dict{LLVM.Value, LLVM.Value}(
        p => LLVM.parameters(new_f)[i] for (i, p) in enumerate(LLVM.parameters(f))
    )
    value_map[f] = new_f
    clone_into!(new_f, f; value_map,
                changes = LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

    # Every load of the builtin becomes a use of the new parameter. The load is
    # what the `llvmcall` in the intrinsic emitted; there is nothing else the
    # global can appear in, since it has no definition to store into.
    for (i, (name, _, _)) in enumerate(used)
        gv = globals(mod)[name]
        arg = LLVM.parameters(new_f)[length(params) + i]
        for use in collect(uses(gv))
            u = user(use)
            u isa LLVM.LoadInst ||
                error("stage builtin @$name is used by $(typeof(u)); only a load is expected")
            replace_uses!(u, arg)
            erase!(u)
        end
    end

    fn = LLVM.name(f)
    GPUCompiler.prune_constexpr_uses!(f)
    @assert isempty(uses(f)) "stage entry $(fn) still has uses after cloning"
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)

    # The globals are dead now; leaving them behind would make the module
    # declare an undefined symbol the loader has to resolve.
    for (name, _, _) in used
        gv = globals(mod)[name]
        isempty(uses(gv)) && erase!(gv)
    end

    return (new_f, Type[m for (_, m, _) in used])
end

"""
    stage_builtins_before_output!(job, mod, f) -> (LLVM.Function, Vector{Type})

`stage_builtins!`, with the output pointer kept in last place.

Builtins are appended at the end, and `stage_return!` takes the LAST parameter
to be the output pointer. Running them in the naive order would leave the output
buried and a builtin holding its place. So the output parameter is moved back to
the end after the append — cheaper and far clearer than teaching
`stage_return!` to look for it somewhere else.
"""
function stage_builtins_before_output!(@nospecialize(job::CompilerJob),
                                       mod::LLVM.Module, f::LLVM.Function)
    new_f, markers = stage_builtins!(job, mod, f)
    isempty(markers) && return (new_f, markers)
    return (rotate_output_last!(job, mod, new_f, length(markers)), markers)
end

"""
    rotate_output_last!(job, mod, f, ntrailing) -> LLVM.Function

Move the parameter sitting `ntrailing` from the end back to the end.

The output pointer was last until `stage_builtins!` appended to the signature.
"""
function rotate_output_last!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                             f::LLVM.Function, ntrailing::Int)
    ft = function_type(f)
    params = collect(LLVM.parameters(ft))
    out_idx = length(params) - ntrailing
    order = vcat(1:(out_idx - 1), (out_idx + 1):length(params), out_idx)

    new_ft = LLVM.FunctionType(LLVM.return_type(ft), LLVMType[params[i] for i in order])
    new_f = LLVM.Function(mod, "", new_ft)
    linkage!(new_f, linkage(f))
    for (new_pos, old_pos) in enumerate(order)
        LLVM.name!(LLVM.parameters(new_f)[new_pos], LLVM.name(LLVM.parameters(f)[old_pos]))
    end

    value_map = Dict{LLVM.Value, LLVM.Value}(
        LLVM.parameters(f)[old_pos] => LLVM.parameters(new_f)[new_pos]
        for (new_pos, old_pos) in enumerate(order)
    )
    value_map[f] = new_f
    clone_into!(new_f, f; value_map,
                changes = LLVM.API.LLVMCloneFunctionChangeTypeGlobalChanges)

    fn = LLVM.name(f)
    GPUCompiler.prune_constexpr_uses!(f)
    @assert isempty(uses(f)) "stage entry $(fn) still has uses after reordering"
    replace_metadata_uses!(f, new_f)
    erase!(f)
    LLVM.name!(new_f, fn)
    return new_f
end
