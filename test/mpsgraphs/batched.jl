using Test, Metal, LinearAlgebra

# MPSGraph work encoded into the OPEN batch, rather than into a command buffer of its
# own.
#
# Two things are silent when they are wrong. Ordering: command buffers run in COMMIT
# order, so an op that commits its own while a batch of launches is still open runs
# BEFORE them — every launch in the batch is ordered against neither side of it. And
# the offset: `MPSGraphTensorData(::MtlArray)` binds the whole buffer and drops
# `arr.offset`, which is correct only for an array that starts one, so a suballocated
# operand reads and writes somebody else's bytes.

const BTG = Metal.MPSGraphs

function bt_setval!(x, v::Float32, n::Int32)
    i = Metal.thread_position_in_grid_1d()
    i <= n && (@inbounds x[i] = v)
    return
end

function bt_trace!(out, c, n::Int32)
    if Metal.thread_position_in_grid_1d() == 1
        t = 0.0f0
        @inbounds for k in Int32(1):n
            t += c[k, k]
        end
        @inbounds out[1] = t
    end
    return
end

"""A `(dims)` array at `off` elements into `big`, which is what a suballocator hands out."""
bt_slice(big, off, dims) = reshape(view(big, (off + 1):(off + prod(dims))), dims)

@testset "MPSGraph joins the batch" begin
    @testset "the product is ordered against the launches around it" begin
        n = 64
        A = MtlArray(zeros(Float32, n, n))
        B = MtlArray(zeros(Float32, n, n))
        C = MtlArray(fill(-1.0f0, n, n))
        tr = MtlArray(zeros(Float32, 1))
        # ONE batch and no synchronisation anywhere inside it: set both operands,
        # multiply, read the product back with a third launch. A product that
        # committed its own command buffer would have run before the two launches
        # that fill its operands and after nothing, giving a product of zeros and a
        # trace read from it.
        Metal.@metal threads=256 groups=cld(n*n, 256) bt_setval!(A, 1.0f0, Int32(n*n))
        Metal.@metal threads=256 groups=cld(n*n, 256) bt_setval!(B, 1.0f0, Int32(n*n))
        BTG.gemm_batched!(C, A, B)
        Metal.@metal threads=1 groups=1 bt_trace!(tr, C, Int32(n))
        Metal.synchronize()
        @test all(==(Float32(n)), Array(C))
        @test Array(tr)[1] == Float32(n * n)
    end

    @testset "`mul!` is ordered the same way" begin
        n = 64
        A = MtlArray(zeros(Float32, n, n))
        B = MtlArray(zeros(Float32, n, n))
        C = MtlArray(fill(-1.0f0, n, n))
        Metal.@metal threads=256 groups=cld(n*n, 256) bt_setval!(A, 2.0f0, Int32(n*n))
        Metal.@metal threads=256 groups=cld(n*n, 256) bt_setval!(B, 1.0f0, Int32(n*n))
        mul!(C, A, B)
        Metal.synchronize()
        @test all(==(2.0f0 * n), Array(C))
    end

    @testset "a suballocated operand is read where it lies" begin
        M, N, K = 64, 48, 32
        Ah = rand(Float32, M, K)
        Bh = rand(Float32, K, N)
        want = Ah * Bh
        big = MtlArray(zeros(Float32, 1 << 18))
        A = bt_slice(big, 1024, (M, K))
        B = bt_slice(big, 1024 + M * K, (K, N))
        C = bt_slice(big, 1024 + M * K + K * N, (M, N))
        copyto!(A, Ah); copyto!(B, Bh)
        @test (A.offset, B.offset, C.offset) != (0, 0, 0)
        BTG.gemm_batched!(C, A, B)
        Metal.synchronize()
        @test Array(C) ≈ want
        # An ignored offset would have written the product at the start of the
        # buffer, over bytes nobody in this product owns.
        @test all(iszero, Array(view(big, 1:1024)))
    end

    @testset "a row bias is folded into the product" begin
        M, N, K = 32, 48, 16
        Ah = rand(Float32, M, K); Bh = rand(Float32, K, N); bh = rand(Float32, M)
        C = MtlArray(zeros(Float32, M, N))
        BTG.gemm_batched!(C, MtlArray(Ah), MtlArray(Bh), MtlArray(bh))
        Metal.synchronize()
        @test Array(C) ≈ Ah * Bh .+ bh
    end

    @testset "which operands the product admits" begin
        ok(dims...) = MtlArray(zeros(Float32, dims...))
        @test BTG.gemm_shape_supported(ok(8, 8), ok(8, 8), ok(8, 8), nothing)
        # A shape no `MPSNDArray` can describe: 3 Float32 is 12 bytes.
        @test !BTG.gemm_shape_supported(ok(3, 8), ok(3, 8), ok(8, 8), nothing)
        # Rank, and a contraction length that does not match.
        @test !BTG.gemm_shape_supported(ok(8, 8, 2), ok(8, 8), ok(8, 8), nothing)
        @test !BTG.gemm_shape_supported(ok(8, 8), ok(8, 4), ok(8, 8), nothing)
        # A bias that is not one value per row.
        @test !BTG.gemm_shape_supported(ok(8, 8), ok(8, 8), ok(8, 8), ok(8, 8))
        @test !BTG.gemm_shape_supported(ok(8, 8), ok(8, 8), ok(8, 8), ok(4))
    end

    if Metal.macos_version() >= v"14"
        @testset "fused attention" begin
            E, Lq, Lk, H, B = 72, 128, 256, 3, 2
            scale = Float32(inv(sqrt(E)))
            qh = Float16.(reshape(0.4 .* sin.(range(0, 9, E*Lq*H*B)), E, Lq, H, B))
            kh = Float16.(reshape(0.4 .* cos.(range(0, 7, E*Lk*H*B)), E, Lk, H, B))
            vh = Float16.(reshape(0.4 .* sin.(range(0, 5, E*Lk*H*B)), E, Lk, H, B))
            o = MtlArray(fill(Float16(NaN), E, Lq, H, B))
            q, k, v = MtlArray(qh), MtlArray(kh), MtlArray(vh)
            @test BTG.sdpa_shape_supported(o, q, k, v)
            BTG.sdpa_batched!(o, q, k, v, scale)
            Metal.synchronize()
            got = Array(o)
            # The destination starts as NaN, so a plane the op skipped cannot pass.
            @test all(isfinite, got)
            ref = zeros(Float64, E, Lq, H, B)
            for b in 1:B, h in 1:H, lq in 1:Lq
                s = [sum(Float64(qh[e,lq,h,b]) * Float64(kh[e,lk,h,b]) for e in 1:E) *
                     Float64(scale) for lk in 1:Lk]
                s .= exp.(s .- maximum(s)); s ./= sum(s)
                for e in 1:E
                    ref[e,lq,h,b] = sum(s[lk] * Float64(vh[e,lk,h,b]) for lk in 1:Lk)
                end
            end
            @test maximum(abs, Float64.(got) .- ref) / maximum(abs, ref) < 3e-3

            # …and which operands it admits.
            mk(T, dims) = MtlArray(zeros(T, dims...))
            @test !BTG.sdpa_shape_supported(mk(Float16, (72, 8)), mk(Float16, (72, 8)),
                                           mk(Float16, (72, 8)), mk(Float16, (72, 8)))
            @test !BTG.sdpa_shape_supported(mk(Float16, (70, 8, 1, 1)),
                                           mk(Float16, (70, 8, 1, 1)),
                                           mk(Float16, (70, 8, 1, 1)),
                                           mk(Float16, (70, 8, 1, 1)))
            @test !BTG.sdpa_shape_supported(mk(Float16, (72, 8, 1, 1)),
                                           mk(Float16, (72, 8, 1, 1)),
                                           mk(Float32, (72, 8, 1, 1)),
                                           mk(Float16, (72, 8, 1, 1)))
        end
    end
end
