using Metal: MtlInlineTensor, TensorOpsMatmul2D, matmul2d_descriptor,
             matmul2d_multiply, matmul2d_multiply_accumulate, MtlCooperativeTensor,
             Int2b, coordinate

# C = A * B sliced over K: a TM x TK slice of A staged in threadgroup memory, B read from the
# device, the sum held in a cooperative destination and stored once, clipped to C.
function coop_gemm_kernel!(C, A, B, ::Val{TM}, ::Val{TN}, ::Val{TK}, ::Val{NSIMD}) where {TM, TN, TK, NSIMD}
    M, K = size(A)
    N = size(B, 2)
    tg = threadgroup_position_in_grid()
    m0 = Int32(tg.x - 1) * Int32(TM)
    n0 = Int32(tg.y - 1) * Int32(TN)
    tid = Int32(thread_index_in_threadgroup())
    sA = MtlThreadGroupArray(Float16, (TM, TK))
    tA = MtlInlineTensor(sA)
    tB = MtlInlineTensor(B)
    op = TensorOpsMatmul2D{matmul2d_descriptor(TM, TN, TK; mode = matmul2d_multiply_accumulate),
                           Int32(NSIMD)}()
    acc = MtlCooperativeTensor{Float32}(op, Float16, Float16)
    ncols = min(Int32(TN), Int32(N) - n0)
    for k0 in Int32(0):Int32(TK):Int32(K - 1)
        i = tid
        while i <= Int32(TM * TK)
            r = (i - Int32(1)) % Int32(TM) + Int32(1)
            c = (i - Int32(1)) ÷ Int32(TM) + Int32(1)
            @inbounds sA[r, c] = m0 + r <= M ? A[m0 + r, k0 + c] : zero(Float16)
            i += Int32(NSIMD * 32)
        end
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        op(tA, view(tB, (k0 + Int32(1), n0 + Int32(1)), (Int32(TK), ncols)), acc)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    end
    tC = MtlInlineTensor(C)
    copyto!(view(tC, (m0 + Int32(1), n0 + Int32(1)), (min(Int32(TM), Int32(M) - m0), ncols)), acc)
    return
end

# One product, then every value through the element interface: C[coordinate] = 2 * acc[i],
# and acc[i] += 1 before the store into D.
function coop_elements_kernel!(C, D, A, B, ::Val{TM}, ::Val{TN}, ::Val{NSIMD}) where {TM, TN, NSIMD}
    op = TensorOpsMatmul2D{matmul2d_descriptor(TM, TN, -1), Int32(NSIMD)}()
    acc = MtlCooperativeTensor{Float32}(op, Float16, Float16)
    op(MtlInlineTensor(A), MtlInlineTensor(B), acc)
    for i in 1:length(acc)
        if isvalid(acc, i)
            r, c = coordinate(acc, i)
            @inbounds C[r, c] = 2 * acc[i]
            acc[i] = acc[i] + 1
        end
    end
    copyto!(MtlInlineTensor(D), acc)
    return
end

# W (M x K) of 2-bit integers, stored K-contiguous and read transposed, times Float16 X.
function coop_int2_kernel!(C, W, X, ::Val{M}, ::Val{N}, ::Val{K}) where {M, N, K}
    op = TensorOpsMatmul2D{matmul2d_descriptor(M, N, K; transpose_left = true), Int32(4)}()
    acc = MtlCooperativeTensor{Float32}(op, Int2b, Float16)
    op(MtlInlineTensor{Int2b}(W, (Int32(K), Int32(M))), MtlInlineTensor(X), acc)
    copyto!(MtlInlineTensor(C), acc)
    return
end

# Two destinations built in a call of their own, over the two halves of B. The storage of a
# destination built in a call that stays one was that call's frame, dead when it returned,
# and every product went into it; a closure `ntuple` calls in a kernel the size of
# DNNKernels' PTQ1 product stayed one. `@noinline` makes it one here, which the compiler
# has to override.
@noinline coop_destination(op) = MtlCooperativeTensor{Float32}(op, Float16, Float16)

function coop_closure_kernel!(C, A, B)
    op = TensorOpsMatmul2D{matmul2d_descriptor(64, 32, -1), Int32(4)}()
    accs = ntuple(_ -> coop_destination(op), Val(2))
    tA, tB, tC = MtlInlineTensor(A), MtlInlineTensor(B), MtlInlineTensor(C)
    K = Int32(size(A, 2))
    for j in 1:2
        c = Int32(32 * (j - 1))
        op(tA, view(tB, (Int32(1), c + Int32(1)), (K, Int32(32))), accs[j])
    end
    for j in 1:2
        c = Int32(32 * (j - 1))
        copyto!(view(tC, (Int32(1), c + Int32(1)), (Int32(64), Int32(32))), accs[j])
    end
    return
end

@testset "cooperative tensors" begin
    if Metal.tensor_matmul_capable()
        # The storage is whatever the back-end says a destination needs; these are shapes
        # where it is NOT M*N*4/threads (TN = 40 and 24 round up, eight simdgroups over a
        # narrow tile reserve twice), so a guessed size corrupts them.
        @testset "sliced $(M)x$(N)x$K, tile $(TM)x$TN, $NSIMD simdgroups" for
                (M, N, K, TM, TN, NSIMD) in ((64, 64, 256, 64, 64, 4), (100, 70, 256, 64, 40, 4),
                                             (64, 48, 128, 64, 24, 4), (64, 32, 128, 32, 64, 8),
                                             (128, 128, 512, 64, 128, 4), (64, 64, 128, 64, 64, 1))
            A = rand(Float16, M, K) .- Float16(0.5)
            B = rand(Float16, K, N) .- Float16(0.5)
            C = MtlArray(fill(NaN32, M, N))
            Metal.@sync @metal threads = 32NSIMD groups = (cld(M, TM), cld(N, TN)) coop_gemm_kernel!(
                C, MtlArray(A), MtlArray(B), Val(TM), Val(TN), Val(32), Val(NSIMD))
            ref = Float32.(A) * Float32.(B)
            @test maximum(abs, Array(C) .- ref) / maximum(abs, ref) < 1.0f-5
        end

        @testset "elements $(TM)x$TN" for (TM, TN) in ((64, 64), (64, 40), (32, 16))
            K = 64
            A = rand(Float16, TM, K) .- Float16(0.5)
            B = rand(Float16, K, TN) .- Float16(0.5)
            C = MtlArray(fill(NaN32, TM, TN))
            D = MtlArray(fill(NaN32, TM, TN))
            Metal.@sync @metal threads = 128 coop_elements_kernel!(C, D, MtlArray(A), MtlArray(B),
                                                                   Val(TM), Val(TN), Val(4))
            ref = Float32.(A) * Float32.(B)
            @test maximum(abs, Array(C) .- 2 .* ref) / maximum(abs, ref) < 1.0f-5
            @test maximum(abs, Array(D) .- (ref .+ 1)) / maximum(abs, ref) < 1.0f-5
        end

        @testset "destinations built in a call of their own" begin
            A = rand(Float16, 64, 64) .- Float16(0.5)
            B = rand(Float16, 64, 64) .- Float16(0.5)
            C = MtlArray(fill(NaN32, 64, 64))
            Metal.@sync @metal threads = 128 coop_closure_kernel!(C, MtlArray(A), MtlArray(B))
            ref = Float32.(A) * Float32.(B)
            @test maximum(abs, Array(C) .- ref) / maximum(abs, ref) < 1.0f-5
        end

        # Four values a byte, the first in the lowest two bits, two's complement.
        @testset "Int2b operand" begin
            M, N, K = 64, 64, 128
            t = rand(Int8(-2):Int8(1), K, M)
            bytes = zeros(UInt8, K * M ÷ 4)
            for (j, v) in enumerate(vec(t))
                bytes[(j - 1) ÷ 4 + 1] |= (UInt8(v & 3) << (2 * ((j - 1) % 4)))
            end
            X = rand(Float16, K, N) .- Float16(0.5)
            C = MtlArray(fill(NaN32, M, N))
            Metal.@sync @metal threads = 128 coop_int2_kernel!(C, MtlArray(bytes), MtlArray(X),
                                                               Val(M), Val(N), Val(K))
            ref = Float32.(permutedims(t)) * Float32.(X)
            @test maximum(abs, Array(C) .- ref) / maximum(abs, ref) < 1.0f-5
        end
    end
end
