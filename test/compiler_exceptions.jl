using Metal, LLVM, Test
using StaticArrays

function exception_tails_fixture(; state = "%state", field = 2, handler = "julia_record_exception_1",
                                   flag = 1, attributes = "", phi = false, cast = true)
    result = phi ? "%result = phi i32 [0, %left], [1, %right]\nret i32 %result" : "ret i32 0"
    acast = cast ? "%ap = bitcast ptr addrspace(1) %a to ptr addrspace(1)" : ""
    bcast = cast ? "%bp = bitcast ptr addrspace(1) %b to ptr addrspace(1)" : ""
    ap = cast ? "%ap" : "%a"
    bp = cast ? "%bp" : "%b"
    return """
    declare fastcc void @julia_record_exception_1(ptr addrspace(1))
    declare fastcc void @julia_record_exception_2(ptr addrspace(1))
    declare void @air.atomic.global.store.i32(ptr addrspace(1), i32, i32, i32, i32, i1)
    define i32 @kernel({i32, ptr addrspace(1), ptr addrspace(1), ptr addrspace(1)} %state,
                       {i32, ptr addrspace(1), ptr addrspace(1), ptr addrspace(1)} %other, i1 %choose) {
    entry:
      br i1 %choose, label %left, label %right
    left:
      %a = extractvalue {i32, ptr addrspace(1), ptr addrspace(1), ptr addrspace(1)} %state, 2
      call fastcc void @julia_record_exception_1(ptr addrspace(1) %a)
      $acast
      call void @air.atomic.global.store.i32(ptr addrspace(1) $ap, i32 1, i32 0, i32 2, i32 0, i1 false)
      br label %exit
    right:
      %b = extractvalue {i32, ptr addrspace(1), ptr addrspace(1), ptr addrspace(1)} $state, $field
      call fastcc void @$handler(ptr addrspace(1) %b) $attributes
      $bcast
      call void @air.atomic.global.store.i32(ptr addrspace(1) $bp, i32 $flag, i32 0, i32 2, i32 0, i1 false)
      br label %exit
    exit:
      $result
    }
    """
end

# Match a shared material's local records: eight 108-byte elements, returned as
# an immutable tuple. The unused array argument of a noinline bounds-error call
# used to keep this MVector alive as an unsupported device heap allocation.
@noinline function checked_local_records(index, value)
    records = MVector{8, NTuple{27, Float32}}(ntuple(i -> ntuple(k -> Float32(i + k), Val(27)), Val(8)))
    records[index] = ntuple(k -> value + Float32(k), Val(27))
    return Tuple(records)
end

function local_records_kernel!(out, indices)
    j = Metal.thread_position_in_grid_1d()
    records = checked_local_records(indices[j], Float32(j))
    result = 0.0f0
    for record in records, value in record
        result += value
    end
    out[j] = result
    return
end

@testset "bounds checks do not make local records escape" begin
    indices = MtlArray(Int32[mod1(j, 8) for j in 1:16])
    out = Metal.zeros(Float32, 16)
    bad = MtlArray(Int32[9])
    try
        @metal threads=16 debug_level=1 local_records_kernel!(out, indices)
        synchronize()
        expected = [sum(sum, checked_local_records(mod1(j, 8), Float32(j))) for j in 1:16]
        @test Array(out) == expected
        @metal threads=1 debug_level=1 local_records_kernel!(out, bad)
        err = try
            synchronize()
            nothing
        catch err
            err
        end
        @test err isa Metal.KernelException
        @test err.name == "BoundsError"
        # A caught failure must leave the device and mailbox usable.
        @metal threads=16 debug_level=1 local_records_kernel!(out, indices)
        synchronize()
        @test Array(out) == expected
    finally
        Metal.unsafe_free!(out)
        Metal.unsafe_free!(indices)
        Metal.unsafe_free!(bad)
    end
end

@testset "identical exception tails" begin
    LLVM.Context() do _
        for (options, merges) in (
                ((;), true), ((; cast = false), true), ((; state = "%other"), false), ((; field = 3), false),
                ((; handler = "julia_record_exception_2"), false), ((; flag = 2), false),
                ((; attributes = "nounwind"), false), ((; phi = true), false),
            )
            mod = parse(LLVM.Module, exception_tails_fixture(; options...))
            try
                @test LLVM.verify(mod) === nothing
                fn = mod.functions["kernel"]
                guard = LLVM.terminator(first(fn.blocks))
                Metal.merge_exception_tails!(mod)
                @test LLVM.verify(mod) === nothing
                @test length(fn.blocks) == (merges ? 3 : 4)
                @test LLVM.terminator(first(fn.blocks)) == guard
                @test length(guard.successors) == 2
                @test (guard.successors[1] == guard.successors[2]) == merges
                once = string(mod)
                Metal.merge_exception_tails!(mod)
                @test string(mod) == once
            finally
                LLVM.dispose(mod)
            end
        end
        # Unknown metadata and address-taken blocks must survive unchanged.
        for address_taken in (false, true)
            mod = parse(LLVM.Module, exception_tails_fixture())
            try
                if address_taken
                    gv = LLVM.GlobalVariable(mod, LLVM.PointerType(), "taken")
                    gv.initializer = LLVM.BlockAddress(mod.functions["kernel"].blocks[2])
                else
                    inst = first(mod.functions["kernel"].blocks[2].instructions)
                    inst.metadata["annotation"] = LLVM.MDNode([LLVM.MDString("keep")])
                end
                before = string(mod)
                Metal.merge_exception_tails!(mod)
                @test LLVM.verify(mod) === nothing
                @test string(mod) == before
            finally
                LLVM.dispose(mod)
            end
        end
        mod = parse(LLVM.Module, replace(exception_tails_fixture(),
            "fastcc void @julia_record_exception_1" => "fastcc i32 @julia_record_exception_1"))
        try
            before = string(mod)
            Metal.merge_exception_tails!(mod)
            @test LLVM.verify(mod) === nothing
            @test string(mod) == before
        finally
            LLVM.dispose(mod)
        end
    end
end

function exception_tail_bounds_kernel!(out, values, choice)
    if choice == 1
        out[1] = values[1]
    elseif choice == 2
        out[1] = values[2]
    else
        out[1] = values[3]
    end
    return
end

@testset "merged tails still report each bounds failure" begin
    out = Metal.zeros(Int32, 1)
    short = MtlArray(Int32[7])
    valid = MtlArray(Int32[7, 11, 13])
    try
        for level in 0:2, choice in 1:3
            @metal threads=1 debug_level=level exception_tail_bounds_kernel!(out, valid, choice)
            synchronize()
            @test only(Array(out)) == (7, 11, 13)[choice]
            choice == 1 && continue
            @metal threads=1 debug_level=level exception_tail_bounds_kernel!(out, short, choice)
            err = try
                synchronize()
                nothing
            catch err
                err
            end
            @test err isa Metal.KernelException
            if level >= 1
                @test err.name == "BoundsError"
                @test err.reason == "Out-of-bounds array access"
            end
            if level == 2
                @test !isempty(err.backtrace)
            end
        end
    finally
        Metal.unsafe_free!(out)
        Metal.unsafe_free!(short)
        Metal.unsafe_free!(valid)
    end
end
