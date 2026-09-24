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
       frag_coord_z, frag_coord_w, frag_coord_xy, dfdx, dfdy,
       candidate_prim_raw, candidate_ox, candidate_oy, candidate_oz,
       candidate_dx, candidate_dy, candidate_dz

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

# ── Screen-space derivatives ─────────────────────────────────────────────────
#
# `dfdx(v)` / `dfdy(v)`: how `v` changes across the fragment quad in x and in y,
# MSL's `dfdx`/`dfdy`. Fragment stage only -- the value is a difference between
# neighbouring invocations, so it exists only where invocations run in a quad.
# AIR gives no diagnostic for calling one elsewhere; the quad simply does not
# exist and the result is undefined rather than an error.
#
# A CALL, unlike everything above: the builtins there are entry parameters that
# `stage_builtins!` appends, but a derivative takes an argument and reads the
# neighbouring invocations, so there is nothing to pass in.
#
# The names are not guessed. There is no `xcrun metal` on this machine, so they
# come from scanning the metallibs Apple ships: `air.dfdx.f32` and `air.dfdy.f32`
# both appear in, among others, IconRendering's and RenderBox's
# `default.metallib`, beside `air.discard_fragment` -- which is the company a
# fragment-only intrinsic should be keeping.
#
# No docstring on either: `@device_function` expands to a CPU stub plus an
# overlay method, and `@doc` on that pair fails to precompile with "cannot
# document the following expression".

@device_function dfdx(v::Float32) =
    @typed_ccall("air.dfdx.f32", llvmcall, Cfloat, (Cfloat,), v)

@device_function dfdy(v::Float32) =
    @typed_ccall("air.dfdy.f32", llvmcall, Cfloat, (Cfloat,), v)

# `discard_fragment()`: MSL's `discard_fragment()`, the fragment-only
# instruction that throws this invocation's colour AND depth away.
#
# `air.discard_fragment` -- no type suffix and no operands, unlike the
# derivatives above. Found the same way, and it is the most abundant graphics
# intrinsic in the system metallibs: dozens of Apple's shipped fragment shaders
# carry it, repeatedly in the company of `air.dfdx.f32`.
#
# Execution CONTINUES past this call -- it is not a terminator, and the shader
# still returns a colour that is then not used. Nothing here marks the call
# `readnone` or `willreturn`, so LLVM treats an unknown external void call as
# side-effecting and neither sinks nor deletes it, which is exactly right: the
# whole point of the call IS its side effect.
@device_function discard_fragment() =
    @typed_ccall("air.discard_fragment", llvmcall, Cvoid, ())

# ── The procedural-candidate builtins ────────────────────────────────────────
#
# Read the same way the stage builtins above are: an external global that
# `stage_builtins!` turns into a trailing parameter of the visible function.
# See `compiler/graphics.jl` for why these are ambient rather than arguments,
# and why each component is its own SCALAR rather than a `float3`.

for (fn, gv) in ((:candidate_prim_raw, "__air_candidate_prim"),)
    @eval @inline function $fn()
        Base.llvmcall(($("""
            @$gv = external global i32
            define i32 @entry() #0 {
                %val = load i32, ptr @$gv, align 4
                ret i32 %val
            }
            attributes #0 = { alwaysinline }
        """), "entry"), UInt32, Tuple{})
    end
end

for (fn, gv) in ((:candidate_ox, "__air_candidate_ox"), (:candidate_oy, "__air_candidate_oy"),
                 (:candidate_oz, "__air_candidate_oz"), (:candidate_dx, "__air_candidate_dx"),
                 (:candidate_dy, "__air_candidate_dy"), (:candidate_dz, "__air_candidate_dz"))
    @eval @inline function $fn()
        Base.llvmcall(($("""
            @$gv = external global float
            define float @entry() #0 {
                %val = load float, ptr @$gv, align 4
                ret float %val
            }
            attributes #0 = { alwaysinline }
        """), "entry"), Float32, Tuple{})
    end
end
