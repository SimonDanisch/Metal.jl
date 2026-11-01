using Metal, LLVM, Test

cache_empty_kernel() = nothing

function cache_live_reference_kernel!(out)
    i = Metal.thread_position_in_grid_1d()
    @inbounds out[i] = UInt(pointer_from_objref(:metal_live_cache_reference))
    return
end

@testset "dead references do not disable persistent caching" begin
    GC = Metal.GPUCompiler
    job = GC.CompilerJob(
        Metal.methodinstance(typeof(cache_empty_kernel), Tuple{}),
        Metal.compiler_config(Metal.device(); name = "cache_dead_reference")
    )
    GC.JuliaContext() do _
        mod, meta = GC.compile(:llvm, job)
        try
            gv = LLVM.GlobalVariable(mod, LLVM.Int64Type(), "unused_symbol_slot")
            gv.initializer = LLVM.ConstantInt(UInt64(0))
            gv.linkage = LLVM.Linkage.Private
            helper = LLVM.Function(mod, "unreachable_symbol_reader", LLVM.FunctionType(LLVM.Int64Type()))
            helper.linkage = LLVM.Linkage.Private
            LLVM.IRBuilder() do builder
                LLVM.position!(builder, LLVM.at_end(LLVM.BasicBlock(helper, "entry")))
                LLVM.ret!(builder, LLVM.load!(builder, LLVM.Int64Type(), gv))
            end
            GC.add_relocation!(meta.relocations,
                GC.Relocation(GC.SlotSite, gv.name, 0, GC.JuliaValueRef(:metal_dead_cache_reference)))
            @test !isempty(meta.relocations)
            @test Metal.persisted_key(job, mod, meta) === nothing
            reference = copy(mod)
            expected = try
                first(GC.emit_asm(job, reference, copy(meta.relocations), LLVM.CodeGenFileType.Object))
            finally
                LLVM.dispose(reference)
            end

            Metal.prune_cache_relocations!(job, mod, meta.relocations)
            @test isempty(meta.relocations)
            @test !haskey(mod.functions, "unreachable_symbol_reader")
            @test !haskey(mod.globals, "unused_symbol_slot")
            @test Metal.persisted_key(job, mod, meta) !== nothing
            actual = first(GC.emit_asm(job, mod, meta.relocations, LLVM.CodeGenFileType.Object))
            @test actual == expected
        finally
            LLVM.dispose(mod)
        end
    end
end

@testset "live references retain the persistent-cache exclusion" begin
    GC = Metal.GPUCompiler
    output = Metal.MtlArray(zeros(UInt, 4))
    try
        tt = Tuple{typeof(Metal.mtlconvert(output))}
        job = GC.CompilerJob(
            Metal.methodinstance(typeof(cache_live_reference_kernel!), tt),
            Metal.compiler_config(Metal.device(); name = "cache_live_reference")
        )
        GC.JuliaContext() do _
            mod, meta = GC.compile(:llvm, job)
            try
                Metal.prune_cache_relocations!(job, mod, meta.relocations)
                @test !isempty(meta.relocations)
                @test Metal.persisted_key(job, mod, meta) === nothing
            finally
                LLVM.dispose(mod)
            end
        end
        Metal.@sync @metal threads = 4 cache_live_reference_kernel!(output)
        @test Array(output) == fill(UInt(pointer_from_objref(:metal_live_cache_reference)), 4)
    finally
        Metal.unsafe_free!(output)
    end
end
