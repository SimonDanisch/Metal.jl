# Stage builtins for graphics shaders: the vertex index, the instance index and
# the interpolated fragment position.
#
# Same names and the same shape as Lava's (`device/gfx_intrinsics.jl`), so a
# shader written once compiles on either backend. Lava reads an external global
# in `addrspace(7)` that its SPIR-V emitter turns into a BuiltIn Input variable;
# these read an external global too, and `compiler/graphics.jl`'s
# `stage_builtins!` turns each one that is actually used into an entry parameter
# carrying the matching `air.*` tag.
#
# The indirection is what lets a builtin be a plain zero-argument call. AIR has
# no such thing as an ambient vertex id — it is a function parameter and nothing
# else — so something has to put it there, and a global the caller cannot see is
# how the shader asks for it without spelling it in its signature.
#
# `vertex_index()` and `instance_index()` return ONE-BASED indices, matching
# Lava, because Julia code indexes from one and a shader that has to remember
# which convention a builtin follows will eventually get it wrong.

export vertex_index, instance_index, frag_coord, frag_coord_x, frag_coord_y,
       frag_coord_z, frag_coord_w, frag_coord_xy

@inline function vertex_index()
    raw = Base.llvmcall(("""
        @__air_stage_vertex_id = external global i32
        define i32 @entry() #0 {
            %val = load i32, ptr @__air_stage_vertex_id, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
    # `raw % Int32`, not `Int32(raw)`: the checked conversion can throw, and a
    # throw reaches `record_exception!`, which reads the KERNEL STATE — which a
    # graphics stage does not have. The value is identical for every vertex id a
    # GPU can produce; only the unreachable error path differs.
    return (raw % Int32) + Int32(1)
end

@inline function instance_index()
    raw = Base.llvmcall(("""
        @__air_stage_instance_id = external global i32
        define i32 @entry() #0 {
            %val = load i32, ptr @__air_stage_instance_id, align 4
            ret i32 %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), UInt32, Tuple{})
    return (raw % Int32) + Int32(1)      # unchecked, as above
end

"""
    frag_coord(dim = 1) -> Float32

One component of the interpolated fragment position, `[[position]]` in MSL.

`dim` is one-based: 1 is x, 4 is w.
"""
@inline function frag_coord(dim::Integer = 1)
    Base.llvmcall(("""
        @__air_stage_frag_coord = external global <4 x float>
        define float @entry(i32 %dim) #0 {
            %vec = load <4 x float>, ptr @__air_stage_frag_coord, align 16
            %val = extractelement <4 x float> %vec, i32 %dim
            ret float %val
        }
        attributes #0 = { alwaysinline }
    """, "entry"), Float32, Tuple{UInt32}, UInt32(dim - 1))
end

@inline frag_coord_x() = frag_coord(1)
@inline frag_coord_y() = frag_coord(2)
@inline frag_coord_z() = frag_coord(3)
@inline frag_coord_w() = frag_coord(4)
@inline frag_coord_xy() = (frag_coord(1), frag_coord(2))
