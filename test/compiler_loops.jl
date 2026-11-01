using Test, Metal, LLVM

function loop_retention_fixture(; metadata = "")
    suffix = isempty(metadata) ? "" : ", !llvm.loop !0"
    return """
    define i32 @sum(i32 %n) {
    entry:
      br label %loop
    loop:
      %i = phi i32 [0, %entry], [%next, %loop]
      %acc = phi i32 [0, %entry], [%sum, %loop]
      %sum = add i32 %acc, %i
      %next = add i32 %i, 1
      %again = icmp ult i32 %next, %n
      br i1 %again, label %loop, label %exit$suffix
    exit:
      ret i32 %sum
    }
    $metadata
    """
end

@testset "native loop-growth control" begin
    @test which(Metal.GPUCompiler.lower_air!, Tuple{Metal.MetalCompilerJob, LLVM.Module}).module === Metal
    LLVM.Context() do ctx
        # Dense functions can cross the instruction limit without crossing the
        # block limit. Exercise its exact boundary independently of block count.
        for dense in (false, true)
            mod = parse(LLVM.Module, loop_retention_fixture())
            try
                instructions = [inst.ref for f in mod.functions for bb in f.blocks for inst in bb.instructions]
                limit = length(instructions) + !dense
                Metal.retain_complex_loops!(mod; min_instructions = limit)
                @test LLVM.verify(mod) === nothing
                @test [inst.ref for f in mod.functions for bb in f.blocks for inst in bb.instructions] == instructions
                @test occursin("llvm.loop.unroll.disable", string(mod)) == dense
                once = string(mod)
                Metal.retain_complex_loops!(mod; min_instructions = limit)
                @test string(mod) == once
            finally
                LLVM.dispose(mod)
            end
        end
        for (name, metadata) in (
                ("none", ""),
                ("vectorize", "!0 = distinct !{!0, !1}\n!1 = !{!\"llvm.loop.vectorize.enable\", i1 true}"),
                ("unroll", "!0 = distinct !{!0, !1}\n!1 = !{!\"llvm.loop.unroll.count\", i32 2}"),
            )
            mod = parse(LLVM.Module, loop_retention_fixture(; metadata))
            try
                before = string(mod)
                instructions = [inst.ref for f in mod.functions for bb in f.blocks for inst in bb.instructions]
                # Production threshold leaves a small kernel completely alone.
                Metal.retain_complex_loops!(mod)
                @test string(mod) == before
                Metal.retain_complex_loops!(mod; min_blocks = 3)
                @test LLVM.verify(mod) === nothing
                @test [inst.ref for f in mod.functions for bb in f.blocks for inst in bb.instructions] == instructions
                lowered = string(mod)
                @test occursin("llvm.loop.unroll.disable", lowered) == (name != "unroll")
                if name == "vectorize"
                    @test occursin("llvm.loop.vectorize.enable", lowered)
                elseif name == "unroll"
                    @test lowered == before
                end
                # Existing hints and unrelated metadata survive repeated passes.
                Metal.retain_complex_loops!(mod; min_blocks = 3)
                @test string(mod) == lowered
            finally
                LLVM.dispose(mod)
            end
        end
    end
end
