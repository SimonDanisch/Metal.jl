# Julia scalar helpers compiled once and reused as native Metal binary functions.
# This intentionally supports only Float32(Float32), without kernel state or
# relocations. It does not yet implement device-side Julia function-table calls,
# aggregate BSDF arguments, or a persistent helper cache.

include("binary_function_reuse.jl")

module JuliaVisibleFunctionProbe
using Metal, LLVM, Test
const GC = Metal.GPUCompiler
const MTL = Metal.MTL

squared(x::Float32) = x*x
doubled(x::Float32) = 2.0f0*x

function library(f, name)
    cfg = Metal.compiler_config(Metal.device(); kernel = false, name, debug_level = 0)
    job = GC.CompilerJob(Metal.methodinstance(typeof(f), Tuple{Float32}), cfg)
    return GC.JuliaContext() do _
        mod, meta = GC.compile(:llvm, job)
        try
            entry = meta.entry
            @test entry.function_type.return_type == LLVM.FloatType()
            @test collect(entry.function_type.parameters) == [LLVM.FloatType()]
            entry.name = name
            # Non-kernel GPUCompiler jobs omit module AIR/Metal version metadata.
            # A standalone visible function still needs it before native compilation.
            GC.add_module_metadata!(job, mod)
            output = LLVM.MDNode([
                LLVM.MDNode(LLVM.Metadata[
                    LLVM.MDString("air.visible_output"),
                    LLVM.MDString("air.arg_type_name"), LLVM.MDString("float")
                ])
            ])
            input = LLVM.MDNode([
                LLVM.MDNode(LLVM.Metadata[
                    LLVM.ConstantInt(Int32(0)), LLVM.MDString("air.visible_input"),
                    LLVM.MDString("air.arg_type_name"), LLVM.MDString("float"),
                    LLVM.MDString("air.arg_name"), LLVM.MDString("x")
                ])
            ])
            push!(get!(LLVM.metadata(mod), "air.visible").operands, LLVM.MDNode(LLVM.Metadata[entry, output, input]))
            LLVM.verify(mod)
            air = first(GC.emit_asm(job, mod, meta.relocations, LLVM.CodeGenFileType.Object))
            fn = Metal.MetalLibFunction(;
                name, air_module = air, program_type = Metal.PROGRAM_VISIBLE,
                air_version = cfg.target.air, metal_version = cfg.target.metal
            )
            lib = Metal.MetalLib(;
                functions = [fn], file_version = Metal.metallib_target(cfg.target.macos),
                platform_version = cfg.target.macos, uuid = Metal.content_uuid(air)
            )
            io = IOBuffer()
            write(io, lib)
            return MTL.MTLLibraryFromData(Metal.device(), take!(io))
        finally
            LLVM.dispose(mod)
        end
    end
end

function run()
    # Reuse the same native binary-function creator and consumers as the MSL control.
    controls = Main.BinaryFunctionReuseProbe
    libs = [library(squared, "julia_visible_squared"), library(doubled, "julia_visible_doubled")]
    helpers = [controls.binary_function(libs[1], "julia_visible_squared"),
               controls.binary_function(libs[2], "julia_visible_doubled")]
    return controls.run(; helpers)
end
end

using Test
@testset "Julia-generated visible Metal functions" begin
    println(JuliaVisibleFunctionProbe.run())
end
