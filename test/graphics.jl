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
