# Julia → AIR vertex and fragment programs.
#
# Metal.jl compiled Julia to compute kernels and nothing else: a `void` entry
# under `air.kernel`, with every scalar argument passed through a buffer. A
# graphics stage differs in three ways that all have to hold at once, so these
# assert the shape of the emitted module rather than any one of them alone.
#
# The reference for every expectation here is a shipping metallib —
# PencilKit's `default.metallib`, 7 vertex and 11 fragment programs — read with
# Metal.jl's own `MetalLib` parser and LLVM.jl. `src/compiler/graphics.jl`
# records what was found and how to reproduce it.

using Test, Metal, LLVM
using Metal: VertexID, InstanceID, FragCoord, mangle_varying, air_stage_name,
             compiler_config, vertex_index, frag_coord_x, frag_coord_y

struct GfxVOut
    position::NTuple{4,Float32}
    color::NTuple{4,Float32}
end

struct GfxFOut
    color::NTuple{4,Float32}
end

function gfx_vertex(verts::Core.LLVMPtr{NTuple{4,Float32},1}, vid::VertexID,
                    out::Core.LLVMPtr{GfxVOut,1})
    p = unsafe_load(verts, Int(vid.value) + 1)
    Base.unsafe_store!(out, GfxVOut(p, (1f0, 0f0, 0f0, 1f0)))
    return nothing
end

function gfx_fragment(pos::FragCoord, out::Core.LLVMPtr{GfxFOut,1})
    Base.unsafe_store!(out, GfxFOut((pos.value[1], pos.value[2], 0f0, 1f0)))
    return nothing
end

"""Compile `f` as `stage` and hand back the LLVM module for inspection."""
function compile_stage(f, tt, stage::Symbol; name = string(nameof(f)))
    cfg = compiler_config(Metal.device(); stage, name)
    job = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(f), tt), cfg)
    return Metal.GPUCompiler.JuliaContext() do _
        Metal.GPUCompiler.compile(:llvm, job)[1]
    end
end

nodes(mod, key) = haskey(LLVM.metadata(mod), key) ?
                  collect(LLVM.operands(LLVM.metadata(mod)[key])) : LLVM.MDNode[]

@testset "the varying linkage tag" begin
    # A varying is matched between stages by THIS STRING and nothing else — not
    # by position, not by name, not by type. These four are what the reference
    # metallib carries; getting the length prefix or the type encoding wrong
    # silently disconnects the stages instead of failing to build.
    @test mangle_varying("snapshotTexCoord", NTuple{2,Float32}) ==
          "generated(16snapshotTexCoordDv2_f)"
    @test mangle_varying("texCoord",   NTuple{2,Float32}) == "generated(8texCoordDv2_f)"
    @test mangle_varying("pixelCoord", NTuple{2,Float32}) == "generated(10pixelCoordDv2_f)"
    @test mangle_varying("color",      NTuple{4,Float32}) == "generated(5colorDv4_f)"

    # `air.arg_type_name` uses a different spelling for the same type, and the
    # reference carries both for every varying.
    @test air_stage_name(NTuple{4,Float32}) == "float4"
    @test air_stage_name(NTuple{2,Float32}) == "float2"
    @test air_stage_name(NTuple{4,Float16}) == "half4"
    @test air_stage_name(UInt32) == "uint"
end

@testset "a Julia function compiles to an AIR vertex program" begin
    mod = compile_stage(gfx_vertex,
                        Tuple{Core.LLVMPtr{NTuple{4,Float32},1}, VertexID,
                              Core.LLVMPtr{GfxVOut,1}},
                        :vertex)

    @test length(nodes(mod, "air.vertex")) == 1
    # …and it must STOP being a kernel: leaving the old node would leave
    # `air.kernel` naming a function that now returns a struct.
    @test isempty(nodes(mod, "air.kernel"))

    entry = only(f for f in LLVM.functions(mod)
                 if !LLVM.isdeclaration(f) && LLVM.name(f) == "gfx_vertex")
    ft = LLVM.function_type(entry)

    # The return type is the whole point. `convert(LLVMType, NTuple{4,Float32})`
    # is `[4 x float]`, an ARRAY; AIR wants `<4 x float>` in a PACKED struct, and
    # the loader rejects the other one.
    rt = LLVM.return_type(ft)
    @test rt isa LLVM.StructType
    @test LLVM.ispacked(rt)
    @test length(LLVM.elements(rt)) == 2
    @test all(e -> e isa LLVM.VectorType, LLVM.elements(rt))
    @test string(rt) == "<{ <4 x float>, <4 x float> }>"

    # The trailing output pointer is gone, and the vertex id is a VALUE — the
    # kernel ABI would have passed it as another buffer pointer.
    params = collect(LLVM.parameters(ft))
    # Compared as text: CONSTRUCTING an LLVM type needs an active context, and
    # the module outlives the one it was compiled in.
    @test string(last(params)) == "i32"

    ops = LLVM.operands(only(nodes(mod, "air.vertex")))
    outs = [string(o) for o in LLVM.operands(ops[2])]
    @test occursin("air.position", outs[1])
    @test occursin("\"position\"", outs[1])
    @test occursin("air.vertex_output", outs[2])
    @test occursin("generated(5colorDv4_f)", outs[2])

    ins = [string(o) for o in LLVM.operands(ops[3])]
    @test any(s -> occursin("air.vertex_id", s), ins)
    # The buffer arguments keep GPUCompiler's description verbatim: a buffer is
    # described the same way whichever stage reads it, so re-deriving sizes and
    # address spaces here would only be a chance to get them wrong.
    @test any(s -> occursin("air.buffer", s) && occursin("air.address_space", s), ins)
end

@testset "a Julia function compiles to an AIR fragment program" begin
    mod = compile_stage(gfx_fragment,
                        Tuple{FragCoord, Core.LLVMPtr{GfxFOut,1}},
                        :fragment)

    @test length(nodes(mod, "air.fragment")) == 1
    @test isempty(nodes(mod, "air.kernel"))

    entry = only(f for f in LLVM.functions(mod)
                 if !LLVM.isdeclaration(f) && LLVM.name(f) == "gfx_fragment")
    @test string(LLVM.return_type(LLVM.function_type(entry))) == "<{ <4 x float> }>"

    ops = LLVM.operands(only(nodes(mod, "air.fragment")))
    # A fragment's outputs are render targets, indexed from zero.
    outs = [string(o) for o in LLVM.operands(ops[2])]
    @test occursin("air.render_target", outs[1])
    @test occursin("i32 0", outs[1])

    # The interpolated position arrives as a value, and declares how it is
    # sampled — the reference pairs `air.position` with `air.center` and
    # `air.no_perspective`.
    ins = [string(o) for o in LLVM.operands(ops[3])]
    pos = only(filter(s -> occursin("air.position", s), ins))
    @test occursin("air.center", pos)
    @test occursin("air.no_perspective", pos)
end

@testset "a kernel is left alone" begin
    # The stage passes must not touch ordinary compute compilation.
    knl(a::Core.LLVMPtr{Float32,1}) = (Base.unsafe_store!(a, 1f0); nothing)
    mod = compile_stage(knl, Tuple{Core.LLVMPtr{Float32,1}}, :kernel)
    @test length(nodes(mod, "air.kernel")) == 1
    @test isempty(nodes(mod, "air.vertex"))
    @test isempty(nodes(mod, "air.fragment"))
    entry = only(f for f in LLVM.functions(mod)
                 if !LLVM.isdeclaration(f) && LLVM.name(f) == "knl")
    @test string(LLVM.return_type(LLVM.function_type(entry))) == "void"
end

@testset "Metal itself accepts the stage" begin
    # The strongest assertion available short of drawing: pack the AIR into a
    # metallib, hand it to the driver, and ask what it thinks it got. Everything
    # above this checks our own emission against a specification; this checks it
    # against Metal.
    #
    # `library.jl` wrote `PROGRAM_KERNEL` for every function it packed, so a
    # stage arrived tagged as a kernel and the loader refused it.
    dev = Metal.device()
    tt  = Tuple{Core.LLVMPtr{NTuple{4,Float32},1}, VertexID, Core.LLVMPtr{GfxVOut,1}}
    cfg = compiler_config(dev; stage = :vertex, name = "gfx_vertex")
    job = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(gfx_vertex), tt), cfg)
    res = Metal.compile_to_metallib(job)

    lib = Metal.MTL.MTLLibraryFromData(dev, res.metallib)
    @test string.(lib.functionNames) == ["gfx_vertex"]
    fn = Metal.MTL.MTLFunction(lib, "gfx_vertex")
    @test fn.functionType == Metal.MTL.MTLFunctionTypeVertex

    # …and a fragment program the same way.
    ftt  = Tuple{FragCoord, Core.LLVMPtr{GfxFOut,1}}
    fcfg = compiler_config(dev; stage = :fragment, name = "gfx_fragment")
    fjob = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(gfx_fragment), ftt), fcfg)
    flib = Metal.MTL.MTLLibraryFromData(dev, Metal.compile_to_metallib(fjob).metallib)
    @test Metal.MTL.MTLFunction(flib, "gfx_fragment").functionType ==
          Metal.MTL.MTLFunctionTypeFragment

    # Note on what was NOT needed: the reader warns "Unknown tag: VATT / VATY"
    # when parsing a shipping metallib — the vertex ATTRIBUTE table and its
    # types. A stage that reads its vertices from a buffer indexed by
    # `[[vertex_id]]`, which is what the reference shader does and what this
    # does, loads without them. They would only be required for a pipeline that
    # fetches attributes through an `MTLVertexDescriptor`.
end

struct GfxOnlyPos
    position::NTuple{4,Float32}
end

function gfx_tri_vertex(verts::Core.LLVMPtr{NTuple{4,Float32},1}, vid::VertexID,
                        out::Core.LLVMPtr{GfxOnlyPos,1})
    Base.unsafe_store!(out, GfxOnlyPos(unsafe_load(verts, Int(vid.value) + 1)))
    return nothing
end
gfx_green(out::Core.LLVMPtr{GfxFOut,1}) =
    (Base.unsafe_store!(out, GfxFOut((0f0, 1f0, 0f0, 1f0))); nothing)

"""Compile `f` as `stage` and hand back the `MTLFunction` the driver sees."""
function stage_function(f, tt, stage::Symbol, name::String)
    dev = Metal.device()
    cfg = compiler_config(dev; stage, name)
    job = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(f), tt), cfg)
    lib = Metal.MTL.MTLLibraryFromData(dev, Metal.compile_to_metallib(job).metallib)
    push!(GFX_LIBS, lib)
    return Metal.MTL.MTLFunction(lib, name)
end

# Held at file scope, not per call. An `MTLFunction` does not keep the library
# it came from alive, and a command queue released while its command buffer is
# still in flight drops the work silently — both show up as an empty target
# rather than as an error.
const GFX_DEV   = Metal.device()
const GFX_QUEUE = Metal.MTL.MTLCommandQueue(GFX_DEV)
const GFX_VERTS = NTuple{4,Float32}[(-0.9f0,-0.9f0,0f0,1f0), (0.9f0,-0.9f0,0f0,1f0),
                                    (0f0,0.9f0,0f0,1f0)]
const GFX_VBUF  = Metal.MTL.MTLBuffer(GFX_DEV, sizeof(GFX_VERTS), pointer(GFX_VERTS);
                                      storage = Metal.SharedStorage)
const GFX_LIBS  = Any[]          # keeps every compiled library reachable

"""Draw one triangle with `vsf`/`fsf` into a 64x64 target and count green pixels."""
function draw_triangle(vsf, fsf)
    MTL = Metal.MTL
    dev = GFX_DEV
    W = H = 64
    td = MTL.MTLTextureDescriptor(MTL.MTLPixelFormatRGBA8Unorm, W, H, false)
    td.usage = MTL.MTLTextureUsageRenderTarget | MTL.MTLTextureUsageShaderRead
    td.storageMode = MTL.MTLStorageModeShared
    tex = MTL.MTLTexture(dev, td)

    pd = MTL.MTLRenderPipelineDescriptor()
    pd.vertexFunction = vsf
    pd.fragmentFunction = fsf
    pd.colorAttachments[1].pixelFormat = MTL.MTLPixelFormatRGBA8Unorm
    pipe = MTL.MTLRenderPipelineState(dev, pd)

    rp = MTL.MTLRenderPassDescriptor()
    ca = rp.colorAttachments[1]
    ca.texture = tex
    ca.loadAction  = MTL.MTLLoadActionClear
    ca.storeAction = MTL.MTLStoreActionStore
    ca.clearColor  = MTL.MTLClearColor(0.0, 0.0, 0.0, 1.0)

    cb  = MTL.MTLCommandBuffer(GFX_QUEUE)
    enc = MTL.MTLRenderCommandEncoder(cb, rp)
    MTL.set_pipeline!(enc, pipe)
    MTL.set_vertex_buffer!(enc, GFX_VBUF, 0, 1)
    MTL.draw_primitives!(enc, MTL.MTLPrimitiveTypeTriangle, 0, 3)
    MTL.endEncoding!(enc)
    MTL.commit!(cb)
    MTL.wait_completed(cb)

    px = Vector{UInt8}(undef, W * H * 4)
    GC.@preserve px MTL.getBytes!(pointer(px), tex, W * 4,
                                  MTL.MTLRegion(MTL.MTLOrigin(0,0,0), MTL.MTLSize(W,H,1)))
    green = count(i -> px[4i + 2] > 0x80, 0:(W*H - 1))
    return (green, px)
end

@testset "a triangle drawn by Julia shaders" begin
    # The end of the chain. Everything above compiles, tags and loads; this is
    # the only test that says the pixels come out right, and it is graded
    # against the same triangle drawn by hand-written MSL rather than against a
    # number somebody wrote down.
    #
    # It is also the test that would have caught the bug the rest could not.
    # `stage_return!` and `stage_inputs!` hand the cloned body a stack `alloca`
    # cast into the parameter's address space — `thread` to `device`, which is
    # not a legal pointer on Metal. The stage still compiled, still linked,
    # still built a render pipeline, and drew NOTHING, because the position was
    # written to a stack slot a device pointer only pretended to name.
    # `stage_cleanup!` runs SROA and the slot goes away; without it this testset
    # reports zero green pixels and every other assertion in this file still
    # passes.
    MTL = Metal.MTL
    dev = Metal.device()

    msl = MTL.MTLLibrary(dev, """
    #include <metal_stdlib>
    using namespace metal;
    struct VO { float4 position [[position]]; };
    vertex VO ref_vs(device const float4 *v [[buffer(0)]], uint vid [[vertex_id]]) {
        VO o; o.position = v[vid]; return o;
    }
    fragment float4 ref_fs() { return float4(0, 1, 0, 1); }
    """)
    ref_vs = MTL.MTLFunction(msl, "ref_vs")
    ref_fs = MTL.MTLFunction(msl, "ref_fs")

    jl_vs = stage_function(gfx_tri_vertex,
                           Tuple{Core.LLVMPtr{NTuple{4,Float32},1}, VertexID,
                                 Core.LLVMPtr{GfxOnlyPos,1}}, :vertex, "gfx_tri_vertex")
    jl_fs = stage_function(gfx_green, Tuple{Core.LLVMPtr{GfxFOut,1}}, :fragment, "gfx_green")

    reference, refpx = draw_triangle(ref_vs, ref_fs)
    # The control really drew a triangle: about 40 % of the frame, which is the
    # area of this one in NDC. A blank or fully-covered target fails here.
    @test 1000 < reference < 3000

    # Each stage on its own against the reference's other half, then both — so a
    # failure names which stage broke instead of just "the image is wrong". The
    # whole buffer is compared, not a sample: the two shaders are supposed to be
    # doing exactly the same arithmetic, so anything but equality is a bug.
    for (label, v, f) in (("Julia vertex", jl_vs, ref_fs),
                          ("Julia fragment", ref_vs, jl_fs),
                          ("both", jl_vs, jl_fs))
        green, px = draw_triangle(v, f)
        @test (label, green) == (label, reference)
        @test (label, px == refpx) == (label, true)
    end
end

# ── the shared shader vocabulary ─────────────────────────────────────────────

struct GfxBuiltinOut
    position::NTuple{4,Float32}
end

# Written the way `Mantle/bench/showcase.jl` writes its shaders: `vertex_index()`
# as a free call, not as a declared parameter. That is Lava's spelling, and the
# point of matching it is that one shader source compiles on either backend.
function gfx_builtin_vertex(verts::Core.LLVMPtr{NTuple{4,Float32},1},
                            out::Core.LLVMPtr{GfxBuiltinOut,1})
    Base.unsafe_store!(out, GfxBuiltinOut(unsafe_load(verts, Int(vertex_index()))))
    return nothing
end

function gfx_builtin_fragment(out::Core.LLVMPtr{GfxFOut,1})
    # Reading the interpolated position is what makes this a stage input rather
    # than a constant-folded shader.
    x = frag_coord_x()
    Base.unsafe_store!(out, GfxFOut((0f0, x > -1f0 ? 1f0 : 0f0, 0f0, 1f0)))
    return nothing
end

@testset "shaders written with the free-function builtins" begin
    # `vertex_index()` is a zero-argument call in Julia and a PARAMETER in AIR —
    # there is no ambient vertex id. `stage_builtins!` bridges that: the
    # intrinsic loads an undefined global, and the pass turns each referenced
    # one into a trailing entry parameter with the matching `air.*` tag.
    #
    # Two things this pins beyond "it compiles":
    #  * the builtin must be APPENDED and the output pointer rotated back to
    #    last, because `stage_return!` takes the last parameter to be the output;
    #  * `vertex_index()` must not use a CHECKED conversion. `Int32(::UInt32)`
    #    can throw, a throw reaches `record_exception!`, that reads the kernel
    #    state, and a graphics stage has none — the compile fails with
    #    "unsupported call to an unknown function (call to julia.gpu.state_getter)".
    jl_vs = stage_function(gfx_builtin_vertex,
                           Tuple{Core.LLVMPtr{NTuple{4,Float32},1},
                                 Core.LLVMPtr{GfxBuiltinOut,1}},
                           :vertex, "gfx_builtin_vertex")
    jl_fs = stage_function(gfx_builtin_fragment, Tuple{Core.LLVMPtr{GfxFOut,1}},
                           :fragment, "gfx_builtin_fragment")

    @test jl_vs.functionType == Metal.MTL.MTLFunctionTypeVertex
    @test jl_fs.functionType == Metal.MTL.MTLFunctionTypeFragment

    green, _ = draw_triangle(jl_vs, jl_fs)
    @test 1000 < green < 3000

    # One-based, like Lava's: a shader that has to remember which convention a
    # builtin follows will eventually get it wrong. If this were zero-based the
    # triangle would read one vertex past its buffer and come out degenerate.
    @test green == first(draw_triangle(
        stage_function(gfx_tri_vertex,
                       Tuple{Core.LLVMPtr{NTuple{4,Float32},1}, VertexID,
                             Core.LLVMPtr{GfxOnlyPos,1}}, :vertex, "gfx_tri_vertex"),
        jl_fs))
end

# ── mesh stages ──────────────────────────────────────────────────────────────
#
# A mesh stage writes its primitives through an object handed to it in address
# space 7, so nothing about it goes through `stage_return!`. Graded against the
# same triangle emitted by hand-written MSL, exactly as the vertex/fragment pair
# above is, because "some pixels came out" is not the assertion — "the same
# pixels the system compiler produces" is.
#
# A mesh stage is a compute stage that rasterises: it has the thread and group
# indices, threadgroup memory and a barrier, and its own buffers numbered from
# zero. The testsets after the drawing one pin each of those, because the
# geometry-to-mesh lowering needs all of them and none of them fails loudly —
# a missing index builtin reads as zero and every thread quietly does the first
# thread's work.

const MeshV = @NamedTuple{position::NTuple{4,Float32}, uv::NTuple{2,Float32}}
const MeshP = @NamedTuple{colour::NTuple{4,Float32}}
const MeshObj = Metal.MeshObject{MeshV, MeshP, 4, 2, :triangle}

# The same fullscreen triangle the MSL reference below emits, vertex for vertex.
function gfx_mesh_tri(out::Metal.MeshPtr{MeshV, MeshP, 4, 2, :triangle})
    Metal.set_position_mesh(out, Int32(0), (-1f0, -1f0, 0f0, 1f0))
    Metal.set_position_mesh(out, Int32(1), ( 3f0, -1f0, 0f0, 1f0))
    Metal.set_position_mesh(out, Int32(2), (-1f0,  3f0, 0f0, 1f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(0), (0f0, 0f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(1), (2f0, 0f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(2), (0f0, 2f0))
    Metal.set_index_mesh(out, Int32(0), UInt8(0))
    Metal.set_index_mesh(out, Int32(1), UInt8(1))
    Metal.set_index_mesh(out, Int32(2), UInt8(2))
    Metal.set_primitive_data_mesh(out, Int32(0), Int32(0), (0f0, 1f0, 0f0, 1f0))
    Metal.set_primitive_count_mesh(out, Int32(1))
    return nothing
end

# The same, with the triangle's extent read out of a buffer of its own — which is
# what every real mesh stage does, and the case the slot rule governs.
function gfx_mesh_scaled(out::Metal.MeshPtr{MeshV, MeshP, 4, 2, :triangle},
                         extent::Core.LLVMPtr{Float32,1})
    s = unsafe_load(extent, 1)
    Metal.set_position_mesh(out, Int32(0), (-s, -s, 0f0, 1f0))
    Metal.set_position_mesh(out, Int32(1), (3f0 * s, -s, 0f0, 1f0))
    Metal.set_position_mesh(out, Int32(2), (-s, 3f0 * s, 0f0, 1f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(0), (0f0, 0f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(1), (2f0, 0f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(2), (0f0, 2f0))
    Metal.set_index_mesh(out, Int32(0), UInt8(0))
    Metal.set_index_mesh(out, Int32(1), UInt8(1))
    Metal.set_index_mesh(out, Int32(2), UInt8(2))
    Metal.set_primitive_data_mesh(out, Int32(0), Int32(0), (0f0, 1f0, 0f0, 1f0))
    Metal.set_primitive_count_mesh(out, Int32(1))
    return nothing
end

# Reads BOTH planes: `uv` is per-vertex and interpolated, `colour` is
# per-primitive. A fragment stage that ignored them would pass even if only one
# plane linked, and the two are matched by separate metadata.
function gfx_mesh_fragment(uv::Metal.Varying{:uv, NTuple{2,Float32}},
                           colour::Metal.Varying{:colour, NTuple{4,Float32}},
                           out::Core.LLVMPtr{GfxFOut,1})
    c = colour.value
    u = uv.value
    Base.unsafe_store!(out, GfxFOut((u[1] * 0f0, c[2], u[2] * 0f0, 1f0)))
    return nothing
end

"""
Draw one mesh threadgroup into a 64x64 target and count green pixels.

`extent` binds a one-float buffer at `slot` when given, which is how the
slot-0 rule is measured rather than asserted.
"""
function draw_mesh(meshfn, fragfn; extent = nothing, slot::Int = 1,
                   threads::Int = 1)
    MTL = Metal.MTL
    dev = GFX_DEV
    W = H = 64

    pd = MTL.MTLMeshRenderPipelineDescriptor()
    pd.meshFunction = meshfn
    pd.fragmentFunction = fragfn
    pd.colorAttachments[1].pixelFormat = MTL.MTLPixelFormatRGBA8Unorm
    pd.maxTotalThreadsPerMeshThreadgroup = threads
    pipe = MTL.MTLRenderPipelineState(dev, pd)

    td = MTL.MTLTextureDescriptor(MTL.MTLPixelFormatRGBA8Unorm, W, H, false)
    td.usage = MTL.MTLTextureUsageRenderTarget | MTL.MTLTextureUsageShaderRead
    td.storageMode = MTL.MTLStorageModeShared
    tex = MTL.MTLTexture(dev, td)

    rp = MTL.MTLRenderPassDescriptor()
    ca = rp.colorAttachments[1]
    ca.texture = tex
    ca.loadAction  = MTL.MTLLoadActionClear
    ca.storeAction = MTL.MTLStoreActionStore
    ca.clearColor  = MTL.MTLClearColor(0.0, 0.0, 0.0, 1.0)

    cb  = MTL.MTLCommandBuffer(GFX_QUEUE)
    enc = MTL.MTLRenderCommandEncoder(cb, rp)
    MTL.set_pipeline!(enc, pipe)
    MTL.set_viewport!(enc, MTL.MTLViewport(0, 0, W, H, 0, 1))
    if extent !== nothing
        # `GC.@preserve` and a named array, not `pointer(Float32[extent])`: the
        # temporary is unreachable the moment `pointer` returns and the copy then
        # reads freed memory. What that produces is a garbage extent, a degenerate
        # triangle and an empty frame — indistinguishable here from the slot bug
        # this testset is about.
        src = Float32[extent]
        GC.@preserve src begin
            buf = MTL.MTLBuffer(dev, sizeof(Float32), pointer(src);
                                storage = Metal.SharedStorage)
        end
        MTL.set_mesh_buffer!(enc, buf, 0, slot)
    end
    MTL.draw_mesh_threadgroups!(enc, MTL.MTLSize(1, 1, 1),
                                     MTL.MTLSize(1, 1, 1), MTL.MTLSize(threads, 1, 1))
    MTL.endEncoding!(enc)
    MTL.commit!(cb)
    MTL.wait_completed(cb)

    px = Vector{UInt8}(undef, W * H * 4)
    GC.@preserve px MTL.getBytes!(pointer(px), tex, W * 4,
                                  MTL.MTLRegion(MTL.MTLOrigin(0,0,0), MTL.MTLSize(W,H,1)))
    green = count(i -> px[4i + 2] > 0x80, 0:(W*H - 1))
    return (green, px)
end

@testset "a mesh stage drawn by Julia shaders" begin
    MTL = Metal.MTL
    dev = Metal.device()

    # Compiled by the system Metal compiler at runtime, so the reference needs no
    # Xcode toolchain — `xcrun metal` is not in the CommandLineTools.
    #
    # A mesh pipeline's fragment stage takes ONE `stage_in` struct nesting both
    # planes. Spelled as two parameters — `(VO v [[stage_in]], PO p)` — it still
    # compiles and still builds a pipeline, and `p` is then an unbound buffer that
    # reads as zeros: a black frame that looks exactly like a mesh stage which
    # emitted nothing.
    msl = MTL.MTLLibrary(dev, """
    #include <metal_stdlib>
    using namespace metal;
    struct VO { float4 position [[position]]; float2 uv; };
    struct PO { float4 colour; };
    using MeshT = metal::mesh<VO, PO, 4, 2, metal::topology::triangle>;
    [[mesh]] void ref_mesh(MeshT out) {
        VO v;
        v.position = float4(-1, -1, 0, 1); v.uv = float2(0, 0); out.set_vertex(0, v);
        v.position = float4( 3, -1, 0, 1); v.uv = float2(2, 0); out.set_vertex(1, v);
        v.position = float4(-1,  3, 0, 1); v.uv = float2(0, 2); out.set_vertex(2, v);
        out.set_index(0, 0); out.set_index(1, 1); out.set_index(2, 2);
        PO p; p.colour = float4(0, 1, 0, 1); out.set_primitive(0, p);
        out.set_primitive_count(1);
    }
    struct FSIn { VO v; PO p; };
    fragment float4 ref_mesh_fs(FSIn in [[stage_in]]) {
        return float4(in.v.uv.x * 0.0, in.p.colour.g, in.v.uv.y * 0.0, 1.0);
    }
    """)
    ref_ms = MTL.MTLFunction(msl, "ref_mesh")
    ref_fs = MTL.MTLFunction(msl, "ref_mesh_fs")
    @test ref_ms.functionType == MTL.MTLFunctionTypeMesh

    jl_ms = stage_function(gfx_mesh_tri, Tuple{Metal.MeshPtr{MeshV, MeshP, 4, 2, :triangle}},
                           :mesh, "gfx_mesh_tri")
    jl_fs = stage_function(gfx_mesh_fragment,
                           Tuple{Metal.Varying{:uv, NTuple{2,Float32}},
                                 Metal.Varying{:colour, NTuple{4,Float32}},
                                 Core.LLVMPtr{GfxFOut,1}}, :fragment, "gfx_mesh_fragment")
    # The driver's own word on what it loaded: a mesh program, not a kernel that
    # happens to be tagged as one.
    @test jl_ms.functionType == MTL.MTLFunctionTypeMesh

    reference, refpx = draw_mesh(ref_ms, ref_fs)
    # A fullscreen triangle covers the frame. Blank fails here, and so does a
    # target that only partly covers because a position went somewhere else.
    @test reference == 64 * 64

    # Each stage against the reference's other half, then both, so a failure names
    # which one broke. Equality of the whole buffer: the two are doing the same
    # arithmetic, and pinning the per-primitive plane needs the exact value.
    for (label, m, f) in (("Julia mesh", jl_ms, ref_fs),
                          ("Julia fragment", ref_ms, jl_fs),
                          ("both", jl_ms, jl_fs))
        green, px = draw_mesh(m, f)
        @test (label, green) == (label, reference)
        @test (label, px == refpx) == (label, true)
    end
end

@testset "a mesh stage's buffers are numbered from zero" begin
    # The object is NOT a buffer and consumes no binding, so a mesh stage's own
    # buffers start at Metal slot 0 exactly like a vertex stage's. MSL agrees:
    # `[[mesh]] void f(MeshT out, device uint *p [[buffer(0)]])` binds at 0.
    #
    # It read the other way round for a while because `retag_stage!` left
    # GPUCompiler's `air.buffer` description of the object in place at location 0
    # and wrote the `air.mesh` node over a different entry. A stage buffer then
    # collided with the object and the draw silently produced nothing.
    jl_ms = stage_function(gfx_mesh_scaled,
                           Tuple{Metal.MeshPtr{MeshV, MeshP, 4, 2, :triangle},
                                 Core.LLVMPtr{Float32,1}}, :mesh, "gfx_mesh_scaled")
    jl_fs = stage_function(gfx_mesh_fragment,
                           Tuple{Metal.Varying{:uv, NTuple{2,Float32}},
                                 Metal.Varying{:colour, NTuple{4,Float32}},
                                 Core.LLVMPtr{GfxFOut,1}}, :fragment, "gfx_mesh_fragment")

    # Slot 1 here is Metal slot 0, and it is where the stage's first buffer goes.
    @test first(draw_mesh(jl_ms, jl_fs; extent = 1f0, slot = 1)) == 64 * 64
    # One past it binds nothing, so the extent reads as whatever is there and the
    # triangle collapses. Pinned because it is the failure the numbering causes.
    @test first(draw_mesh(jl_ms, jl_fs; extent = 1f0, slot = 2)) == 0

    # And the buffer is genuinely read rather than the shader having constants
    # folded in: a smaller extent covers strictly fewer pixels.
    full    = first(draw_mesh(jl_ms, jl_fs; extent = 1f0,    slot = 1))
    half    = first(draw_mesh(jl_ms, jl_fs; extent = 0.5f0,  slot = 1))
    quarter = first(draw_mesh(jl_ms, jl_fs; extent = 0.25f0, slot = 1))
    @test full > half > quarter > 0
end

# ── the data intrinsics take the FIELD before the SLOT ───────────────────────
#
# `air.set_vertex_data_mesh` and `air.set_primitive_data_mesh` were emitted with
# their two i32 operands the other way round, matching `air.set_position_mesh`,
# and every test above agreed with the mistake. The two orders OVERLAP: a stage
# writing field 0 of vertices 0..n calls `(0,0) (1,0) (2,0)`, which read as
# `(field, slot)` writes field 0 of vertex 0 and then fields 1 and 2 of vertex 0,
# which do not exist. So the FIRST vertex is right and the rest silently vanish,
# and a shader that samples one vertex — or reads only the per-primitive plane,
# which is what every testset above does — cannot tell.
#
# This one can: four vertices with four different `uv`s and two triangles with two
# different per-primitive colours, graded against the same thing in MSL. Under the
# old order the interpolated `uv` collapsed to a ramp falling away from vertex 0
# and the second triangle came out black.

function gfx_mesh_quad(out::Metal.MeshPtr{MeshV, MeshP, 4, 2, :triangle})
    Metal.set_position_mesh(out, Int32(0), (-1f0, -1f0, 0f0, 1f0))
    Metal.set_position_mesh(out, Int32(1), ( 1f0, -1f0, 0f0, 1f0))
    Metal.set_position_mesh(out, Int32(2), (-1f0,  1f0, 0f0, 1f0))
    Metal.set_position_mesh(out, Int32(3), ( 1f0,  1f0, 0f0, 1f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(0), (0f0, 0f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(1), (1f0, 0f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(2), (0f0, 1f0))
    Metal.set_vertex_data_mesh(out, Int32(0), Int32(3), (1f0, 1f0))
    Metal.set_index_mesh(out, Int32(0), UInt8(0))
    Metal.set_index_mesh(out, Int32(1), UInt8(1))
    Metal.set_index_mesh(out, Int32(2), UInt8(2))
    Metal.set_index_mesh(out, Int32(3), UInt8(2))
    Metal.set_index_mesh(out, Int32(4), UInt8(1))
    Metal.set_index_mesh(out, Int32(5), UInt8(3))
    Metal.set_primitive_data_mesh(out, Int32(0), Int32(0), (0f0, 1f0, 0f0, 1f0))
    Metal.set_primitive_data_mesh(out, Int32(0), Int32(1), (0f0, 1f0, 1f0, 1f0))
    Metal.set_primitive_count_mesh(out, Int32(2))
    return nothing
end

# `uv` in red and green, the per-primitive blue in blue: one fragment stage that
# shows both planes at once, and shows WHICH vertex and which triangle each value
# came from.
function gfx_mesh_quad_fragment(uv::Metal.Varying{:uv, NTuple{2,Float32}},
                                colour::Metal.Varying{:colour, NTuple{4,Float32}},
                                out::Core.LLVMPtr{GfxFOut,1})
    u = uv.value
    c = colour.value
    Base.unsafe_store!(out, GfxFOut((u[1], u[2], c[3], 1f0)))
    return nothing
end

@testset "a mesh stage's data planes are addressed per field and per slot" begin
    MTL = Metal.MTL
    dev = Metal.device()

    msl = MTL.MTLLibrary(dev, """
    #include <metal_stdlib>
    using namespace metal;
    struct VO { float4 position [[position]]; float2 uv; };
    struct PO { float4 colour; };
    using MeshT = metal::mesh<VO, PO, 4, 2, metal::topology::triangle>;
    [[mesh]] void ref_quad(MeshT out) {
        VO v;
        v.position = float4(-1, -1, 0, 1); v.uv = float2(0, 0); out.set_vertex(0, v);
        v.position = float4( 1, -1, 0, 1); v.uv = float2(1, 0); out.set_vertex(1, v);
        v.position = float4(-1,  1, 0, 1); v.uv = float2(0, 1); out.set_vertex(2, v);
        v.position = float4( 1,  1, 0, 1); v.uv = float2(1, 1); out.set_vertex(3, v);
        out.set_index(0, 0); out.set_index(1, 1); out.set_index(2, 2);
        out.set_index(3, 2); out.set_index(4, 1); out.set_index(5, 3);
        PO p;
        p.colour = float4(0, 1, 0, 1); out.set_primitive(0, p);
        p.colour = float4(0, 1, 1, 1); out.set_primitive(1, p);
        out.set_primitive_count(2);
    }
    struct FSIn { VO v; PO p; };
    fragment float4 ref_quad_fs(FSIn in [[stage_in]]) {
        return float4(in.v.uv.x, in.v.uv.y, in.p.colour.b, 1.0);
    }
    """)
    ref_ms = MTL.MTLFunction(msl, "ref_quad")
    ref_fs = MTL.MTLFunction(msl, "ref_quad_fs")

    jl_ms = stage_function(gfx_mesh_quad,
                           Tuple{Metal.MeshPtr{MeshV, MeshP, 4, 2, :triangle}},
                           :mesh, "gfx_mesh_quad")
    jl_fs = stage_function(gfx_mesh_quad_fragment,
                           Tuple{Metal.Varying{:uv, NTuple{2,Float32}},
                                 Metal.Varying{:colour, NTuple{4,Float32}},
                                 Core.LLVMPtr{GfxFOut,1}}, :fragment,
                           "gfx_mesh_quad_fragment")

    _, refpx = draw_mesh(ref_ms, ref_fs)
    _, jlpx  = draw_mesh(jl_ms, jl_fs)

    # What the reference itself has to show, so this testset says what it means
    # even when the two agree on something wrong: `uv` sweeps nearly the whole
    # range in both directions, and the two triangles differ in blue. Nearly,
    # because a pixel is sampled at its CENTRE and the outermost centre is half a
    # pixel inside the quad — the exact extremes are the rasteriser's business and
    # not what is being pinned.
    reds   = [refpx[4i + 1] for i in 0:(64 * 64 - 1)]
    greens = [refpx[4i + 2] for i in 0:(64 * 64 - 1)]
    blues  = [refpx[4i + 3] for i in 0:(64 * 64 - 1)]
    @test minimum(reds)   < 0x08 && maximum(reds)   > 0xf0
    @test minimum(greens) < 0x08 && maximum(greens) > 0xf0
    @test Set(unique(blues)) == Set([0x00, 0xff])
    # About half the quad each: the diagonal splits it, and which side the pixels
    # ON the diagonal land in is a fill rule.
    @test count(==(0xff), blues) ≈ count(==(0x00), blues) rtol = 0.05

    # …and then the whole frame, byte for byte. Under the old operand order the
    # reds and greens collapsed and every blue was 0x00.
    @test jlpx == refpx
end


# ── a mesh stage is a compute stage that rasterises ──────────────────────────

const MeshProbeObj = Metal.MeshPtr{MeshV, MeshP, 4, 2, :triangle}

"""
Every thread records its own indices, so a collapsed index cannot hide.

It also emits a triangle, because a mesh stage that emits NOTHING is dropped
whole once the fragment stage reads its planes — side effects and all. That is
not a bug to work around; it is why the probe has to be a real drawing stage.
"""
function gfx_mesh_indices(out::MeshProbeObj, probe::Core.LLVMPtr{UInt32,1})
    tid = Metal.thread_index_in_threadgroup()          # 1-based
    gid = Metal.threadgroup_position_in_grid().x       # 1-based
    slot = (Int(gid) - 1) * 8 + Int(tid)
    Base.unsafe_store!(probe, UInt32(gid) * UInt32(100) + UInt32(tid), slot)
    Base.unsafe_store!(probe, Metal.threads_per_threadgroup().x, 33)
    if tid == UInt32(1)
        Metal.set_position_mesh(out, Int32(0), (-1f0, -1f0, 0f0, 1f0))
        Metal.set_position_mesh(out, Int32(1), ( 3f0, -1f0, 0f0, 1f0))
        Metal.set_position_mesh(out, Int32(2), (-1f0,  3f0, 0f0, 1f0))
        Metal.set_vertex_data_mesh(out, Int32(0), Int32(0), (0f0, 0f0))
        Metal.set_vertex_data_mesh(out, Int32(0), Int32(1), (2f0, 0f0))
        Metal.set_vertex_data_mesh(out, Int32(0), Int32(2), (0f0, 2f0))
        Metal.set_index_mesh(out, Int32(0), UInt8(0))
        Metal.set_index_mesh(out, Int32(1), UInt8(1))
        Metal.set_index_mesh(out, Int32(2), UInt8(2))
        Metal.set_primitive_data_mesh(out, Int32(0), Int32(0), (0f0, 1f0, 0f0, 1f0))
        Metal.set_primitive_count_mesh(out, Int32(1))
    end
    return nothing
end

"""
Four threads meet in threadgroup memory: three write a corner, one reads all
three back and emits the triangle. A barrier that does not hold, or threadgroup
memory a mesh stage cannot have, gives a degenerate triangle and no coverage.
"""
function gfx_mesh_cooperative(out::MeshProbeObj)
    corners = Metal.MtlThreadGroupArray(NTuple{4,Float32}, 4)
    tid = Metal.thread_index_in_threadgroup()
    x = tid == UInt32(1) ? -1f0 : (tid == UInt32(2) ?  3f0 : -1f0)
    y = tid == UInt32(1) ? -1f0 : (tid == UInt32(2) ? -1f0 :  3f0)
    tid <= UInt32(3) && (@inbounds corners[tid] = (x, y, 0f0, 1f0))
    Metal.threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if tid == UInt32(1)
        @inbounds Metal.set_position_mesh(out, Int32(0), corners[1])
        @inbounds Metal.set_position_mesh(out, Int32(1), corners[2])
        @inbounds Metal.set_position_mesh(out, Int32(2), corners[3])
        Metal.set_vertex_data_mesh(out, Int32(0), Int32(0), (0f0, 0f0))
        Metal.set_vertex_data_mesh(out, Int32(0), Int32(1), (2f0, 0f0))
        Metal.set_vertex_data_mesh(out, Int32(0), Int32(2), (0f0, 2f0))
        Metal.set_index_mesh(out, Int32(0), UInt8(0))
        Metal.set_index_mesh(out, Int32(1), UInt8(1))
        Metal.set_index_mesh(out, Int32(2), UInt8(2))
        Metal.set_primitive_data_mesh(out, Int32(0), Int32(0), (0f0, 1f0, 0f0, 1f0))
        Metal.set_primitive_count_mesh(out, Int32(1))
    end
    return nothing
end

"""Run `meshfn` over `groups` x `threads` and hand back the probe buffer."""
function run_mesh_probe(meshfn, fragfn, groups::Int, threads::Int)
    MTL = Metal.MTL
    dev = GFX_DEV
    pd = MTL.MTLMeshRenderPipelineDescriptor()
    pd.meshFunction = meshfn
    pd.fragmentFunction = fragfn
    pd.colorAttachments[1].pixelFormat = MTL.MTLPixelFormatRGBA8Unorm
    pd.maxTotalThreadsPerMeshThreadgroup = threads
    pipe = MTL.MTLRenderPipelineState(dev, pd)

    # The probe emits no primitive, so the fragment stage never runs; it is here
    # because a colour attachment without one is not a pipeline Metal will build.
    n = 40
    buf = MTL.MTLBuffer(dev, n * sizeof(UInt32); storage = Metal.SharedStorage)
    ptr = convert(Ptr{UInt32}, MTL.contents(buf))
    for i in 1:n
        Base.unsafe_store!(ptr, typemax(UInt32), i)
    end

    td = MTL.MTLTextureDescriptor(MTL.MTLPixelFormatRGBA8Unorm, 8, 8, false)
    td.usage = MTL.MTLTextureUsageRenderTarget
    td.storageMode = MTL.MTLStorageModeShared
    rp = MTL.MTLRenderPassDescriptor()
    ca = rp.colorAttachments[1]
    ca.texture = MTL.MTLTexture(dev, td)
    ca.loadAction = MTL.MTLLoadActionClear
    ca.storeAction = MTL.MTLStoreActionStore
    ca.clearColor = MTL.MTLClearColor(0.0, 0.0, 0.0, 1.0)

    cb  = MTL.MTLCommandBuffer(GFX_QUEUE)
    enc = MTL.MTLRenderCommandEncoder(cb, rp)
    MTL.set_pipeline!(enc, pipe)
    MTL.set_mesh_buffer!(enc, buf, 0, 1)
    MTL.draw_mesh_threadgroups!(enc, MTL.MTLSize(groups, 1, 1),
                                     MTL.MTLSize(1, 1, 1), MTL.MTLSize(threads, 1, 1))
    MTL.endEncoding!(enc)
    MTL.commit!(cb)
    MTL.wait_completed(cb)
    return [Base.unsafe_load(ptr, i) for i in 1:n]
end

@testset "a mesh stage gets the compute builtins" begin
    # The geometry-to-mesh lowering runs one thread per input vertex and one
    # invocation per input primitive, so it needs every thread to know which it
    # is. When the index builtins are missing they do not fail — they read zero,
    # every thread does the first thread's work, and the stage behaves like a
    # single-threaded one that happens to write the same slots repeatedly.
    jl_ms = stage_function(gfx_mesh_indices,
                           Tuple{MeshProbeObj, Core.LLVMPtr{UInt32,1}},
                           :mesh, "gfx_mesh_indices")
    jl_fs = stage_function(gfx_mesh_fragment,
                           Tuple{Metal.Varying{:uv, NTuple{2,Float32}},
                                 Metal.Varying{:colour, NTuple{4,Float32}},
                                 Core.LLVMPtr{GfxFOut,1}}, :fragment, "gfx_mesh_fragment")
    r = run_mesh_probe(jl_ms, jl_fs, 3, 4)

    # Twelve invocations, each with its own (group, thread), and gid*100+tid is
    # unique per pair — so a collapsed index shows up as a missing slot.
    seen = [(g, t, r[(g - 1) * 8 + t]) for g in 1:3, t in 1:4]
    @test all(x -> x[3] == UInt32(x[1] * 100 + x[2]), seen)
    @test length(unique(x[3] for x in seen)) == 12
    @test r[33] == UInt32(4)      # threads_per_threadgroup, from the descriptor
end

@testset "a mesh stage gets threadgroup memory and a barrier" begin
    jl_ms = stage_function(gfx_mesh_cooperative, Tuple{MeshProbeObj},
                           :mesh, "gfx_mesh_cooperative")
    jl_fs = stage_function(gfx_mesh_fragment,
                           Tuple{Metal.Varying{:uv, NTuple{2,Float32}},
                                 Metal.Varying{:colour, NTuple{4,Float32}},
                                 Core.LLVMPtr{GfxFOut,1}}, :fragment, "gfx_mesh_fragment")
    @test first(draw_mesh(jl_ms, jl_fs; threads = 4)) == 64 * 64
end


# ── Sampling a bound texture ─────────────────────────────────────────────────
#
# The AGX compiler recognises a texture argument by its POINTEE's struct name, and
# LLVM 22 has no pointee to give it. `compiler/texture.jl` records how it is put
# back and what the alternative was: handed `{} addrspace(1)*`, the compiler
# service SEGFAULTS, and `MTLRenderPipelineState` reports it as
# `XPC_ERROR_CONNECTION_INTERRUPTED` while naming nothing. Both halves are pinned
# here — the shape of the AIR, and a pipeline actually built from it.

function gfx_tex_fragment(pos::FragCoord, tex::Metal.Texture2DPtr{Float32},
                          samp::Metal.SamplerPtr, out::Core.LLVMPtr{GfxFOut,1})
    s = Metal.air_sample_texture_2d(tex, samp, 0.25f0, 0.25f0)
    Base.unsafe_store!(out, GfxFOut((s.value[1].value, s.value[2].value,
                                     s.value[3].value, 1f0)))
    return nothing
end

const GFX_TEX_TT = Tuple{FragCoord, Metal.Texture2DPtr{Float32}, Metal.SamplerPtr,
                         Core.LLVMPtr{GfxFOut,1}}

"""The AIR a stage is handed to the driver as, disassembled at bitcode 14."""
function stage_air_text(f, tt, stage::Symbol, name::String)
    cfg = compiler_config(Metal.device(); stage, name)
    job = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(f), tt), cfg)
    bytes = Metal.compile_to_metallib(job).metallib
    fn = only(read(IOBuffer(bytes), Metal.MetalLib).functions)
    # Parsed in process, not disassembled by `llvm-dis`. The tool was reached as
    # `LLVMDowngrader_jll.llvm_dis_14()`, which that package does not export on
    # every version it resolves to — and an `UndefVarError` inside a testset
    # ABORTS THE FILE, so every testset after this one silently did not run.
    # `LLVM.jl` is already a dependency here and already parses this module.
    LLVM.@dispose ctx = LLVM.Context() begin
        return string(parse(LLVM.Module, fn.air_module))
    end
end

@testset "a texture argument reaches AIR as an opaque texture pointer" begin
    air = stage_air_text(gfx_tex_fragment, GFX_TEX_TT, :fragment, "gfx_tex_fragment")

    # The two handle types, BODYLESS. A struct with a body — `type {}` or
    # `type { i8 }` — puts the crash straight back; only `opaque` builds.
    @test occursin("%struct._texture_2d_t = type opaque", air)
    @test occursin("%struct._sampler_t = type opaque", air)

    entry = only(filter(l -> startswith(l, "define"), split(air, '\n')))
    # OPAQUE-POINTER spelling: `ptr addrspace(1) byref(%struct._texture_2d_t)`.
    # These read `%struct._texture_2d_t addrspace(1)*` when the text came from
    # `llvm-dis-14`, which still printed typed pointers. The PROPERTY is the
    # same and is what matters — the parameter is a pointer to the texture
    # handle type, in the texture address space, and the pointee is still named.
    @test occursin("ptr addrspace(1) byref(%struct._texture_2d_t)", entry)
    @test occursin("ptr addrspace(2) byref(%struct._sampler_t)", entry)
    # …and NOT a pointee it could not work out, which is what the parameters
    # degrade to the moment the `byref` is lost — the crash this pins.
    @test !occursin("byref({})", entry)
    @test occursin("byref(", entry)

    # The argument metadata names them as a texture and a sampler rather than as
    # the buffers the kernel ABI made them, each with its own location namespace.
    @test occursin("!\"air.texture\"", air)
    @test occursin("!\"air.sampler\"", air)
    @test occursin("texture2d<float, sample>", air)
end

@testset "a fragment stage samples a bound texture" begin
    MTL = Metal.MTL
    dev = GFX_DEV
    # 2x2, one distinct colour per texel. (0.25, 0.25) is the centre of texel
    # (0, 0) with nearest filtering, so the whole triangle takes its colour.
    texels = UInt8[0x00, 0xff, 0x00, 0xff,   0xff, 0x00, 0x00, 0xff,
                   0x00, 0x00, 0xff, 0xff,   0xff, 0xff, 0x00, 0xff]
    td = MTL.MTLTextureDescriptor(MTL.MTLPixelFormatRGBA8Unorm, 2, 2, false)
    td.usage = MTL.MTLTextureUsageShaderRead
    td.storageMode = MTL.MTLStorageModeShared
    tex = MTL.MTLTexture(dev, td)
    GC.@preserve texels MTL.replace_region!(
        tex, MTL.MTLRegion(MTL.MTLOrigin(0, 0, 0), MTL.MTLSize(2, 2, 1)), 0,
        convert(Ptr{Cvoid}, pointer(texels)), 2 * 4)

    sd = MTL.MTLSamplerDescriptor()
    sd.minFilter = MTL.MTLSamplerMinMagFilterNearest
    sd.magFilter = MTL.MTLSamplerMinMagFilterNearest
    samp = MTL.MTLSamplerState(dev, sd)

    vsf = stage_function(gfx_vertex,
                         Tuple{Core.LLVMPtr{NTuple{4,Float32},1}, VertexID,
                               Core.LLVMPtr{GfxVOut,1}}, :vertex, "gfx_vertex")
    fsf = stage_function(gfx_tex_fragment, GFX_TEX_TT, :fragment, "gfx_tex_fragment")

    # Building the pipeline state is the step that used to kill the compiler
    # service, so reaching the draw at all is half the assertion.
    W = H = 64
    rtd = MTL.MTLTextureDescriptor(MTL.MTLPixelFormatRGBA8Unorm, W, H, false)
    rtd.usage = MTL.MTLTextureUsageRenderTarget | MTL.MTLTextureUsageShaderRead
    rtd.storageMode = MTL.MTLStorageModeShared
    target = MTL.MTLTexture(dev, rtd)

    pd = MTL.MTLRenderPipelineDescriptor()
    pd.vertexFunction = vsf
    pd.fragmentFunction = fsf
    pd.colorAttachments[1].pixelFormat = MTL.MTLPixelFormatRGBA8Unorm
    pipe = MTL.MTLRenderPipelineState(dev, pd)

    rp = MTL.MTLRenderPassDescriptor()
    ca = rp.colorAttachments[1]
    ca.texture = target
    ca.loadAction  = MTL.MTLLoadActionClear
    ca.storeAction = MTL.MTLStoreActionStore
    ca.clearColor  = MTL.MTLClearColor(0.0, 0.0, 0.0, 1.0)

    cb  = MTL.MTLCommandBuffer(GFX_QUEUE)
    enc = MTL.MTLRenderCommandEncoder(cb, rp)
    MTL.set_pipeline!(enc, pipe)
    MTL.set_vertex_buffer!(enc, GFX_VBUF, 0, 1)
    # 1-based, like every other binder here: the wrapper subtracts one.
    MTL.set_fragment_texture!(enc, tex, 1)
    MTL.set_fragment_sampler!(enc, samp, 1)
    MTL.draw_primitives!(enc, MTL.MTLPrimitiveTypeTriangle, 0, 3)
    MTL.endEncoding!(enc)
    MTL.commit!(cb)
    MTL.wait_completed(cb)

    px = Vector{UInt8}(undef, W * H * 4)
    GC.@preserve px MTL.getBytes!(pointer(px), target, W * 4,
                                  MTL.MTLRegion(MTL.MTLOrigin(0,0,0), MTL.MTLSize(W,H,1)))
    # The texel at (0, 0) is green, and the triangle covers the middle of the
    # target — so a covered pixel carries the SAMPLE and not the clear colour.
    mid = ((H ÷ 2) * W + (W ÷ 2)) * 4
    @test px[mid + 1] == 0x00 && px[mid + 2] == 0xff && px[mid + 3] == 0x00
    @test count(i -> px[4i + 2] > 0x80, 0:(W*H - 1)) > 500
end

# ── A VISIBLE function ───────────────────────────────────────────────────────

struct VisOut
    v::NTuple{4,Float32}
end

# The shape a procedural ray-tracing candidate needs — `(prim, o, d, best)` in,
# `(hit, t, a, b)` out. Written like a fragment stage, with a trailing output
# pointer, because that is what `stage_return!` turns into a real AIR return.
function vis_candidate(prim::UInt32, ox::Float32, oy::Float32, oz::Float32,
                       dx::Float32, dy::Float32, dz::Float32, best::Float32,
                       out::Core.LLVMPtr{VisOut,1})
    t = ox * dx + oy * dy + oz * dz + Float32(prim)
    Base.unsafe_store!(out, VisOut((t < best ? 1f0 : 0f0, t, 0.25f0, 0.5f0)))
    return nothing
end

const VIS_TT = Tuple{UInt32, Float32, Float32, Float32, Float32, Float32, Float32,
                     Float32, Core.LLVMPtr{VisOut,1}}

"""What `vis_candidate` computes, on the host."""
vis_ref(i) = (1f0, 1f0*0.5f0 + 2f0*0.25f0 + 3f0*0.125f0 + Float32(i), 0.25f0, 0.5f0)

@testset "a Julia function compiles to an AIR visible function" begin
    # Why this exists: `metal::raytracing::intersector<>` and
    # `intersection_query<>` are C++ class templates the Metal frontend
    # instantiates and inlines, so a traversal LOOP can only be MSL — while the
    # body that has to run for a procedural box (Hikari's Newton solve) is
    # Julia. A `[[visible]]` function is the join: MSL runs the query, this runs
    # the solve, and only scalars cross.
    dev = Metal.device()
    cfg = compiler_config(dev; stage = :visible, name = "vis_candidate")
    job = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(vis_candidate), VIS_TT), cfg)
    res = Metal.compile_to_metallib(job)

    # The metallib TAG. `library.jl` hardcoded `PROGRAM_KERNEL` for everything it
    # packed; a visible function packed under that tag is not one.
    lib = read(IOBuffer(res.metallib), Metal.MetalLib)
    @test only(lib.functions).program_type == Metal.PROGRAM_VISIBLE

    mod = compile_stage(vis_candidate, VIS_TT, :visible; name = "vis_candidate")

    # BY VALUE, and returning the value BARE. Both are the ABI a caller links
    # against and neither is what a stage gets: the kernel ABI passes every
    # argument as `ptr addrspace(1)`, and every other stage returns a packed
    # struct because it has several outputs to name. Left either way this still
    # compiles, links, dispatches and completes — and the caller reads zeros.
    @test string(LLVM.function_type(LLVM.functions(mod)["vis_candidate"])) ==
          "<4 x float> (i32, float, float, float, float, float, float, float)"

    # …and the AIR metadata, whose shape is Apple's: `air.visible` holding
    # `{ptr @fn, outputs, inputs}`, the output carrying only a TYPE — no index
    # and no name, unlike a render target or a varying. Read out of
    # `CC_InlineCompositing32x32` in CoreComposite's shipped `default-cc.metallib`.
    vis = nodes(mod, "air.visible")
    @test length(vis) == 1
    ops = collect(LLVM.operands(vis[1]))
    @test length(ops) == 3
    outs = collect(LLVM.operands(ops[2]))
    @test length(outs) == 1
    @test occursin("air.visible_output", string(outs[1]))
    @test occursin("float4", string(outs[1]))
    ins = collect(LLVM.operands(ops[3]))
    @test length(ins) == 8
    @test all(i -> occursin("air.visible_input", string(ins[i])), 1:length(ins))
    # `uint`, from the JULIA type: LLVM cannot tell it from `int` — both are
    # `i32` — and Apple's own visible functions spell the unsigned one `uint`.
    @test occursin("!\"uint\"", string(ins[1]))
    @test occursin("!\"float\"", string(ins[2]))

    # It is no longer a kernel: leaving `air.kernel` behind would point it at a
    # function that now returns a value.
    @test isempty(nodes(mod, "air.kernel"))

    # …and Metal agrees.
    vislib = Metal.MTL.MTLLibraryFromData(dev, res.metallib)
    visfn  = Metal.MTL.MTLFunction(vislib, "vis_candidate")
    @test visfn.functionType == Metal.MTL.MTLFunctionTypeVisible
end

@testset "an MSL kernel calls the Julia visible function" begin
    # END TO END, and the point of the whole exercise.
    #
    # Through a TABLE and not an `extern` declaration linked by name: the
    # frontend `newLibraryWithSource:` runs resolves symbols immediately and
    # refuses an unresolved one ("Undefined symbol(s) for architecture
    # 'air64'"). An AIR module can carry an unresolved extern — which is how a
    # Julia kernel calls INTO MSL — but a source-compiled MSL kernel cannot, so
    # this direction indexes a visible function table instead.
    dev = Metal.device()
    cfg = compiler_config(dev; stage = :visible, name = "vis_candidate")
    job = Metal.GPUCompiler.CompilerJob(Metal.methodinstance(typeof(vis_candidate), VIS_TT), cfg)
    visfn = Metal.MTL.MTLFunction(
        Metal.MTL.MTLLibraryFromData(dev, Metal.compile_to_metallib(job).metallib),
        "vis_candidate")

    caller = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void call_vis(
        device float4 *out [[buffer(0)]],
        visible_function_table<float4(uint, float, float, float, float, float, float, float)> tbl [[buffer(1)]],
        uint tid [[thread_position_in_grid]])
    {
        out[tid] = tbl[0](tid, 1.0f, 2.0f, 3.0f, 0.5f, 0.25f, 0.125f, 100.0f);
    }
    """
    cfn = Metal.MTL.MTLFunction(Metal.MTL.MTLLibrary(dev, caller), "call_vis")
    desc = Metal.MTLComputePipelineDescriptor()
    desc.computeFunction = cfn
    lf = Metal.MTL.MTLLinkedFunctions()
    # `functions`, not `privateFunctions`: a table entry is reached through a
    # pointer, so the function has to survive as a real call with a stable ABI
    # rather than being inlined into this pipeline.
    lf.functions = Metal.NSArray([visfn])
    desc.linkedFunctions = lf
    desc.maxCallStackDepth = 4
    pipe = Metal.MTLComputePipelineState(dev, desc)

    # `nothing` here means the function was not linked in — worth its own
    # assertion, because an unset table entry is a GPU fault and not an error.
    handle = Metal.MTL.function_handle(pipe, visfn)
    @test handle !== nothing
    tbl = Metal.MTL.MTLVisibleFunctionTable(pipe, Metal.MTL.MTLVisibleFunctionTableDescriptor(1))
    Metal.MTL.set_function!(tbl, handle, 0)

    n = 8
    out = Metal.MtlVector{NTuple{4,Float32}}(undef, n)
    cb  = Metal.MTL.MTLCommandBuffer(GFX_QUEUE)
    enc = Metal.MTL.MTLComputeCommandEncoder(cb)
    Metal.MTL.set_function!(enc, pipe)
    Metal.MTL.set_buffer!(enc, out.data[], 0, 1)
    Metal.MTL.set_visible_function_table!(enc, tbl, 1)
    Metal.MTL.append_current_function!(enc, Metal.MTL.MTLSize(1,1,1), Metal.MTL.MTLSize(n,1,1))
    Metal.MTL.endEncoding!(enc)
    Metal.MTL.commit!(cb)
    Metal.MTL.wait_completed(cb)

    # Bit-identical to the same function on the host — not merely non-zero. A
    # wrong ABI reads as all zeros, which `!= 0` would catch, but a SHIFTED one
    # reads as plausible garbage and would not.
    @test Array(out) == [vis_ref(i) for i in 0:(n - 1)]
end
