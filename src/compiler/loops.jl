# Apple's native compiler can exhaust memory while unrolling a very large
# control-flow graph. A 5,309-block material kernel aborted in native codegen
# with report_bad_alloc_error; retaining its loops compiled in 5.3 seconds.
# Marking only its six largest loop bodies was insufficient. Keep loops in
# exceptionally large functions, without changing the policy for small kernels
# or overriding an explicit source-level unroll request.
# Dense graphs need the same policy: a 3,922-block shader timed out at 90 seconds
# without it and compiled in 3.7 seconds with it. Its 49,766 LLVM instructions
# expand to 67,879 during AIR downgrade; this threshold is applied before that.
function retain_complex_loops!(mod::LLVM.Module; min_blocks::Int = 4096, min_instructions::Int = 32768)
    function large(f)
        length(f.blocks) >= min_blocks && return true
        instructions = 0
        for bb in f.blocks, _ in bb.instructions
            instructions += 1
            instructions >= min_instructions && return true
        end
        return false
    end
    function retain(loop)
        latches = [LLVM.terminator(bb) for bb in LLVM.predecessors(loop.header) if bb in loop]
        nodes = [term.metadata["llvm.loop"] for term in latches if haskey(term.metadata, "llvm.loop")]
        options = isempty(nodes) ? LLVM.Metadata[] : collect(first(nodes).operands)[2:end]
        explicit = any(options) do option
            option isa LLVM.MDNode || return false
            any(option.operands) do operand
                operand isa LLVM.MDString && startswith(String(operand), "llvm.loop.unroll.")
            end
        end
        if !explicit
            id = LLVM.TemporaryMDNode() do temp
                disable = LLVM.MDNode([LLVM.MDString("llvm.loop.unroll.disable")])
                LLVM.replace_temporary!(temp, LLVM.MDNode([temp.node; options; disable]))
            end
            for term in latches
                term.metadata["llvm.loop"] = id
            end
        end
        foreach(retain, loop.subloops)
        return nothing
    end
    pass = LLVM.FunctionPass(
        "metal_retain_complex_loops", (f, am) -> begin
            large(f) || return false
            foreach(retain, am[LLVM.LoopInfo])
            # Only optimization metadata changed; the instructions and CFG did not.
            false
        end; analyses = true
    )
    LLVM.run!(pass, mod)
    return mod
end
