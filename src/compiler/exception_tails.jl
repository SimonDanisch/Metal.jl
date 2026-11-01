# Hundreds of identical exception tails can make Apple's native compiler spend
# minutes on an otherwise small kernel. Keep every guard, but share the complete
# reporting/flag/exit block. Match the lowered tail narrowly: no escaping values,
# PHIs, operand bundles, call-site attributes or metadata other than source lines.
function exception_tail_key(bb::LLVM.BasicBlock)
    insts = collect(bb.instructions)
    length(insts) in (4, 5) || return nothing
    state, report = insts[1:2]
    signal, exit = insts[(end - 1):end]
    cast = length(insts) == 5 ? insts[3] : state
    state isa LLVM.ExtractValueInst && report isa LLVM.CallInst &&
        (cast == state || cast isa LLVM.BitCastInst) && signal isa LLVM.CallInst &&
        exit isa LLVM.BrInst || return nothing
    length(exit.successors) == 1 || return nothing
    report.called_operand isa LLVM.Function || return nothing
    startswith(LLVM.name(report.called_operand), "julia_record_exception_") || return nothing
    signal.called_operand isa LLVM.Function || return nothing
    LLVM.name(signal.called_operand) == "air.atomic.global.store.i32" || return nothing
    report.value_type isa LLVM.VoidType && signal.value_type isa LLVM.VoidType || return nothing
    length(report.arguments) == 1 && only(report.arguments) == state || return nothing
    cast == state || only(cast.operands) == state || return nothing
    first(signal.arguments) == cast || return nothing
    for inst in insts
        all(kv -> first(kv) == LLVM.MD_dbg, inst.metadata) || return nothing
    end
    for call in (report, signal)
        isempty(call.function_attributes) && isempty(call.return_attributes) &&
            all(isempty, call.argument_attributes) && isempty(call.operand_bundles) || return nothing
        call.tailcall && return nothing
    end
    # Deleting the block must not delete a value used by a surviving instruction.
    for inst in (state, cast), user in inst.users
        user isa LLVM.Instruction && user.parent == bb || return nothing
    end
    # Shared successors with PHIs would require updating their incoming edges.
    target = only(exit.successors)
    first(target.instructions) isa LLVM.PHIInst && return nothing
    all(user -> user isa Union{LLVM.BrInst, LLVM.SwitchInst}, bb.users) || return nothing
    return (
        only(state.operands), Tuple(state.indices), state.value_type, cast.value_type,
        report.called_operand, report.called_type, report.callconv,
        signal.called_operand, signal.called_type, signal.callconv,
        Tuple(signal.arguments)[2:end], target,
    )
end

function merge_exception_tails!(mod::LLVM.Module)
    for f in mod.functions
        tails = Dict{Any, LLVM.BasicBlock}()
        for bb in collect(f.blocks)
            key = exception_tail_key(bb)
            key === nothing && continue
            if haskey(tails, key)
                LLVM.replace_uses!(bb, tails[key])
                LLVM.erase!(bb)
            else
                tails[key] = bb
            end
        end
    end
    return mod
end
