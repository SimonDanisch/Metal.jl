# Julia → AIR mesh programs.
#
# The specification is in `compiler/graphics.jl`'s mesh section, read out of
# `particle_gaussian_mesh` in VFX.framework's shipping metallib. What this file
# adds is the emission, and it differs from the vertex and fragment stages in
# exactly one structural way: the entry stays `void` and everything it produces
# goes through an OBJECT it is handed as its first parameter, in address space 7.
#
# So `stage_return!` does not run for a mesh stage — there is no trailing output
# pointer to turn into a return value — and instead the object's argument
# metadata is retagged from the `air.buffer` GPUCompiler described it as.
#
# ── air.mesh_type_info ───────────────────────────────────────────────────────
#
# The object's type, and the reason `MeshConfig`'s bounds are compile-time
# constants:
#
#   !{"air.mesh_type_info", <vertex info>, <primitive info>,
#     i32 <max_vertices>, i32 <max_primitives>, "air.triangle"}
#
# `<vertex info>` is one node per field of the per-vertex struct, the first of
# which is the clip position:
#
#   {air.position, air.arg_type_name, "float4", air.arg_name, "position"}
#   {air.mesh_vertex_data, i32 <n>, "generated(<mangled>)",
#    air.arg_type_name, "float2", air.arg_name, "uv0"}
#
# `<primitive info>` is one node per field of the per-primitive struct, all of
# the second shape with `air.mesh_primitive_data`.
#
# The `generated(...)` string is what `mangle_varying` already produces for the
# vertex and fragment stages — verified against the shipping metallib, which
# spells `crworld_position::float3` as `generated(16crworld_positionDv3_f)`. So
# a fragment stage reading a mesh stage's outputs links by exactly the string it
# already used, and needs nothing new.

"""
    MeshObject{V, P, NV, NP, Topo}

The output object a mesh stage writes through, as a type.

`V` is the per-vertex NamedTuple (its first field must be `position`), `P` the
per-primitive one, `NV`/`NP` the bounds, and `Topo` a `Symbol` naming what the
emitted primitives are. Phantom: the value is a pointer to it, and every field
here is something a compiler must know before it can emit the entry at all.
"""
struct MeshObject{V, P, NV, NP, Topo} end

"""
    MeshPtr{V, P, NV, NP, Topo}

The first parameter of a Julia mesh stage: a pointer to its output object in
address space 7, which is the one AIR gives a mesh object.
"""
const MeshPtr{V, P, NV, NP, Topo} = Core.LLVMPtr{MeshObject{V, P, NV, NP, Topo}, 7}

"""What AIR calls a mesh stage's output topology."""
air_mesh_topology(t::Symbol) =
    t === :triangle ? "air.triangle" :
    t === :line     ? "air.line" :
    t === :point    ? "air.point" :
    error("no AIR mesh topology for :$t; expected :triangle, :line or :point")

# One node per field of the per-vertex struct. `position` is `air.position` and
# carries no index; every other field is numbered from zero, in declaration
# order, which is the contract the fragment stage's inputs are matched against.
function air_mesh_vertex_info(@nospecialize(V::Type))
    names = fieldnames(V)
    isempty(names) && error("a mesh stage's vertex struct needs at least a position")
    first(names) === :position ||
        error("the first field of a mesh stage's vertex struct must be `position`, got `$(first(names))`")
    out = Metadata[]
    for (i, name) in enumerate(names)
        T = fieldtype(V, name)
        if i == 1
            push!(out, MDNode(Metadata[MDString("air.position"),
                                       MDString("air.arg_type_name"),
                                       MDString(air_stage_name(T)),
                                       MDString("air.arg_name"),
                                       MDString("position")]))
        else
            push!(out, MDNode(Metadata[MDString("air.mesh_vertex_data"),
                                       Metadata(ConstantInt(Int32(i - 2))),
                                       MDString(mangle_varying(string(name), T)),
                                       MDString("air.arg_type_name"),
                                       MDString(air_stage_name(T)),
                                       MDString("air.arg_name"),
                                       MDString(string(name))]))
        end
    end
    return out
end

# One node per field of the per-primitive struct — the `Flat`-declared outputs.
# An empty struct is a real answer: a stage whose every output varies across the
# primitive has no per-primitive plane.
function air_mesh_primitive_info(@nospecialize(P::Type))
    out = Metadata[]
    for (i, name) in enumerate(fieldnames(P))
        T = fieldtype(P, name)
        push!(out, MDNode(Metadata[MDString("air.mesh_primitive_data"),
                                   Metadata(ConstantInt(Int32(i - 1))),
                                   MDString(mangle_varying(string(name), T)),
                                   MDString("air.arg_type_name"),
                                   MDString(air_stage_name(T)),
                                   MDString("air.arg_name"),
                                   MDString(string(name))]))
    end
    return out
end

"""
    air_mesh_type_name(V, P, NV, NP, Topo) -> String

How AIR spells the object's type, which the shipping metallib gives as
`mesh<particle_vertex_io, particle_primitive_io, 96, 32, triangle>`.

The two struct names are ours to choose — nothing matches on them, the
`generated(...)` strings inside carry the linkage — so they say what they are.
"""
air_mesh_type_name(NV::Integer, NP::Integer, Topo::Symbol) =
    "mesh<mesh_vertex_io, mesh_primitive_io, $NV, $NP, $Topo>"

"""
    air_mesh_argument(T) -> Metadata

The argument entry for a mesh stage's output object, which replaces the
`air.buffer` description GPUCompiler emitted for parameter one.
"""
function air_mesh_argument(@nospecialize(T::Type))
    T <: MeshObject ||
        error("the first argument of a mesh stage must point at a MeshObject, got $T")
    V, P, NV, NP, Topo = T.parameters
    info = MDNode(Metadata[MDString("air.mesh_type_info"),
                           MDNode(air_mesh_vertex_info(V)),
                           MDNode(air_mesh_primitive_info(P)),
                           Metadata(ConstantInt(Int32(NV))),
                           Metadata(ConstantInt(Int32(NP))),
                           MDString(air_mesh_topology(Topo))])
    return MDNode(Metadata[Metadata(ConstantInt(Int32(0))),
                           MDString("air.mesh"),
                           info,
                           MDString("air.arg_type_name"),
                           MDString(air_mesh_type_name(NV, NP, Topo)),
                           MDString("air.arg_name"),
                           MDString("output")])
end

"""
    mesh_object_type(job) -> Type

The `MeshObject` this mesh stage writes through, read off the first argument.

Derived rather than carried in `MetalCompilerParams` for the reason
`stage_output_type` is: the argument the body writes through IS the declaration
of what the stage produces, and a second source of truth is a second thing to
get out of sync.
"""
function mesh_object_type(@nospecialize(job::CompilerJob))
    args = job.source.specTypes.parameters[2:end]
    isempty(args) &&
        error("a mesh stage needs its output object as its first argument; the signature has none")
    P = first(args)
    P <: Core.LLVMPtr ||
        error("the first argument of a mesh stage must be a `MeshPtr`, got $P")
    return P.parameters[1]
end

# ── The intrinsics ───────────────────────────────────────────────────────────
#
# All five take the object as their first argument, which is why an intrinsic
# that took only a slot could never be lowered here: in AIR the object is a
# PARAMETER of the entry, so there is nothing global to reach.
#
#   air.set_position_mesh          (ptr addrspace(7), i32, <4 x float>)
#   air.set_vertex_data_mesh.<T>   (ptr addrspace(7), i32, i32, <T>)
#   air.set_primitive_data_mesh.<T>(ptr addrspace(7), i32, i32, <T>)
#   air.set_index_mesh             (ptr addrspace(7), i32, i8)
#   air.set_primitive_count_mesh   (ptr addrspace(7), i32)
#
# Slots are ZERO-based here — this is the AIR boundary, and one-based is the
# convention above it. `Mantle`'s overrides subtract.

"""The `<T>` suffix AIR gives a mesh data intrinsic for value type `T`."""
air_mesh_suffix(::Type{Float32}) = "f32"
air_mesh_suffix(::Type{Float16}) = "f16"
air_mesh_suffix(::Type{Int32})   = "i32"
air_mesh_suffix(::Type{Int16})   = "i16"
air_mesh_suffix(::Type{UInt32})  = "i32"
air_mesh_suffix(::Type{UInt16})  = "i16"
air_mesh_suffix(::Type{NTuple{N,T}}) where {N,T} = "v$(N)" * air_mesh_suffix(T)
function air_mesh_suffix(@nospecialize(T::Type))
    V = air_wrapped_vector(T)
    V === nothing &&
        error("no AIR mesh data suffix for $T; a mesh output field must be a " *
              "scalar or a small vector of one")
    return air_mesh_suffix(V)
end

"""An LLVM vector, which is what AIR's `<N x T>` operands are."""
@inline air_vec(v::NTuple{N,T}) where {N,T} = ntuple(i -> VecElement(v[i]), Val(N))
@inline air_vec(v) = air_vec(air_unwrap_vector(v))

"""The tuple inside a one-field vector wrapper such as `Vec4f`."""
@inline air_unwrap_vector(v::NTuple{N,T}) where {N,T} = v
@inline air_unwrap_vector(v) = Base.getfield(v, 1)

# set_position_mesh(out, slot, position)
#
# The clip position of the vertex at `slot`, counting from ZERO. Its own
# intrinsic rather than field zero of the vertex data, which is what the shipping
# metallib shows.
#
# A comment and not a docstring: `@device_function` expands into a CPU stub and
# an overlay method, and Julia cannot attach one docstring to that pair.
@device_function @inline function set_position_mesh(out::Core.LLVMPtr{T,7}, slot::Int32,
                                                    position) where {T}
    @typed_ccall("air.set_position_mesh", llvmcall, Nothing,
                 (Core.LLVMPtr{T,7}, Int32, NTuple{4,VecElement{Float32}}),
                 out, slot, air_vec(position))
end

# set_index_mesh(out, slot, vertexindex)
#
# The vertex index at index-slot `slot`, both counting from ZERO.
#
# `vertexindex` is written as an i8: a mesh output object addresses at most 256
# vertices, which is a bound the caller has to respect and `MeshConfig` refuses
# to exceed.
@device_function @inline function set_index_mesh(out::Core.LLVMPtr{T,7}, slot::Int32,
                                                 vertexindex::UInt8) where {T}
    @typed_ccall("air.set_index_mesh", llvmcall, Nothing,
                 (Core.LLVMPtr{T,7}, Int32, UInt8), out, slot, vertexindex)
end

# set_primitive_count_mesh(out, n)
#
# How many primitives this threadgroup filled. Called once per threadgroup.
@device_function @inline function set_primitive_count_mesh(out::Core.LLVMPtr{T,7},
                                                          n::Int32) where {T}
    @typed_ccall("air.set_primitive_count_mesh", llvmcall, Nothing,
                 (Core.LLVMPtr{T,7}, Int32), out, n)
end
