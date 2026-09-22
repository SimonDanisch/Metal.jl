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

"""`sdpa_batched!` through the same question a backend asks; `nothing` if declined."""
function bt_sdpa!(o, q, k, v, scale)
    ops = BTG.sdpa_operands(o, q, k, v)
    ops === nothing && return nothing
    BTG.sdpa_batched!(o, ops.res..., ops.keys..., scale)
end

"""`softmax(scale·QᵀK)·V` per `(head, batch)` plane, in Float64."""
function bt_attn_ref(qh, kh, vh, scale)
    E, Lq, H, B = size(qh); Lk = size(kh, 2)
    out = zeros(Float64, E, Lq, H, B)
    for b in 1:B, h in 1:H, lq in 1:Lq
        s = [sum(Float64(qh[e, lq, h, b]) * Float64(kh[e, lk, h, b]) for e in 1:E) * scale
             for lk in 1:Lk]
        s .= exp.(s .- maximum(s)); s ./= sum(s)
        for e in 1:E
            out[e, lq, h, b] = sum(s[lk] * Float64(vh[e, lk, h, b]) for lk in 1:Lk)
        end
    end
    out
end

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

    # `matmul2d` accumulates in Float32 and stores once, which is why a caller may
    # ask for a Float32 destination from Float16 operands and get the accumulator it
    # wanted. MPSGraph's product has no compute type — its output dtype follows its
    # inputs — so the same ask is a HALF product cast afterwards. Taking it moved SAM
    # 2.1's encoder `add_129` from 0.48 to 1.12, past the band the Vulkan reference
    # sits in, and nothing about the result looked wrong.
    @testset "a destination wider than the operands is refused" begin
        half(dims...) = MtlArray(zeros(Float16, dims...))
        wide(dims...) = MtlArray(zeros(Float32, dims...))
        @test BTG.gemm_shape_supported(half(64, 64), half(64, 64), half(64, 64), nothing)
        @test !BTG.gemm_shape_supported(wide(64, 64), half(64, 64), half(64, 64), nothing)
        @test BTG.gemm_shape_supported(wide(64, 64), wide(64, 64), wide(64, 64), nothing)
    end

    @testset "the activation on the store" begin
        # A&S 7.1.26, in Float64 where its 1.5e-7 is far below what fp16 can show.
        function erf_ref(x::Float64)
            a = abs(x)
            t = 1.0 / (1.0 + 0.3275911a)
            y = 1.0 - (((((1.061405429t - 1.453152027)t + 1.421413741)t -
                         0.284496736)t + 0.254829592)t) * exp(-a * a)
            x < 0 ? -y : y
        end
        gelu_ref(x) = 0.5x * (1 + erf_ref(x / sqrt(2.0)))
        gelutanh_ref(x) = 0.5x * (1 + tanh(0.7978845608028654 * (x + 0.044715x^3)))
        # The product is `xs[i]` in every column, so the activation is exercised over
        # the whole range — including below -2, where `1 + erf` cancels and a half
        # evaluation of it keeps a couple of significant bits.
        Mw, Nw, Kw = 512, 8, 16
        Ah = zeros(Float16, Mw, Kw); Bh = zeros(Float16, Kw, Nw)
        xs = Float32.(range(-8, 8; length = Mw))
        Ah[:, 1] .= Float16.(xs); Bh[1, :] .= Float16(1)
        A, B = MtlArray(Ah), MtlArray(Bh)
        for (act, ref) in ((:identity, identity), (:relu, x -> max(x, 0.0)),
                           (:gelu, gelu_ref), (:gelu_tanh, gelutanh_ref))
            C = MtlArray(fill(Float16(NaN), Mw, Nw))
            BTG.gemm_batched!(C, A, B, nothing, act)
            Metal.synchronize()
            got = Float64.(Array(C)[:, 1])
            want = ref.(Float64.(xs))
            @test all(isfinite, got)
            # Half's own rounding at this scale, and nothing more: an activation
            # evaluated in half rather than single is an order of magnitude worse.
            @test maximum(abs, got .- want) / maximum(abs, want) < 1e-3
        end
        @test_throws ArgumentError BTG.gemm_batched!(
            MtlArray(zeros(Float16, Mw, Nw)), A, B, nothing, :softplus)
    end

    @testset "which operands the product admits" begin
        ok(dims...) = MtlArray(zeros(Float32, dims...))
        @test BTG.gemm_shape_supported(ok(8, 8), ok(8, 8), ok(8, 8), nothing)
        # An activation the graph has no node for is not a shape it admits.
        @test !BTG.gemm_shape_supported(ok(8, 8), ok(8, 8), ok(8, 8), nothing, :softplus)
        @test BTG.gemm_shape_supported(ok(8, 8), ok(8, 8), ok(8, 8), nothing, :gelu)
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
            @test bt_sdpa!(o, q, k, v, scale) !== nothing
            Metal.synchronize()
            got = Array(o)
            # The destination starts as NaN, so a plane the op skipped cannot pass.
            @test all(isfinite, got)
            ref = bt_attn_ref(qh, kh, vh, Float64(scale))
            @test maximum(abs, Float64.(got) .- ref) / maximum(abs, ref) < 3e-3

            # …and which operands it admits.
            mk(T, dims) = MtlArray(zeros(T, dims...))
            @test BTG.sdpa_operands(mk(Float16, (72, 8)), mk(Float16, (72, 8)),
                                    mk(Float16, (72, 8)), mk(Float16, (72, 8))) === nothing
            @test BTG.sdpa_operands(mk(Float16, (70, 8, 1, 1)), mk(Float16, (70, 8, 1, 1)),
                                    mk(Float16, (70, 8, 1, 1)),
                                    mk(Float16, (70, 8, 1, 1))) === nothing
            @test BTG.sdpa_operands(mk(Float16, (72, 8, 1, 1)), mk(Float16, (72, 8, 1, 1)),
                                    mk(Float32, (72, 8, 1, 1)),
                                    mk(Float16, (72, 8, 1, 1))) === nothing
        end

        # The RESULT type is asked separately from the operand type. A graph that
        # projects in fp16 and declares an fp32 result is an ordinary export -- it is
        # what Qwen-Image 2.1 does -- and requiring all four to agree declined 32 of
        # its attention ops into the three-pass path, ~85 s of a 144 s step.
        #
        # The wider destination is a WIDENED fp16 answer and not an fp32 accumulator:
        # the op's result dtype follows its inputs. That is pinned below, because it
        # is the one thing about this gate somebody will assume the other way round,
        # and because it is exactly the property that makes the matmul gate refuse
        # the same pair. What must NOT be admitted is q, k and v disagreeing with
        # EACH OTHER, which the testset above pins.
        @testset "the result type need not be the operand type" begin
            E, Lq, Lk, H, B = 72, 64, 96, 2, 2
            scale = Float32(inv(sqrt(E)))
            qh = Float16.(reshape(0.4 .* sin.(range(0, 9, E*Lq*H*B)), E, Lq, H, B))
            kh = Float16.(reshape(0.4 .* cos.(range(0, 7, E*Lk*H*B)), E, Lk, H, B))
            vh = Float16.(reshape(0.4 .* sin.(range(0, 5, E*Lk*H*B)), E, Lk, H, B))
            q, k, v = MtlArray(qh), MtlArray(kh), MtlArray(vh)
            ref = bt_attn_ref(qh, kh, vh, Float64(scale))

            # NaN rather than zeros: a plane the op skipped reads as an answer
            # otherwise, which is how a read-before-write hides.
            o32 = MtlArray(fill(Float32(NaN), E, Lq, H, B))
            @test bt_sdpa!(o32, q, k, v, scale) !== nothing
            o16 = MtlArray(fill(Float16(NaN), E, Lq, H, B))
            @test bt_sdpa!(o16, q, k, v, scale) !== nothing
            Metal.synchronize()
            g32, g16 = Array(o32), Array(o16)
            @test all(isfinite, g32)
            @test all(isfinite, g16)
            e32 = maximum(abs, Float64.(g32) .- ref) / maximum(abs, ref)
            e16 = maximum(abs, Float64.(g16) .- ref) / maximum(abs, ref)
            @test e32 < 3e-3
            # The SAME number, not a smaller one: the op computed in fp16 either
            # way and the fp32 destination holds that answer widened. Pinned as an
            # equality so that a future MPSGraph which really does accumulate in
            # single fails here and gets the docstring corrected, rather than
            # quietly making a `<=` look prescient.
            @test e32 == e16

            # fp32 operands with an fp16 destination is the same door, the other way,
            # and there the operand type is what carries the precision: computed in
            # single and narrowed once at the end, it beats the fp16 operands above.
            o = MtlArray(fill(Float16(NaN), E, Lq, H, B))
            @test bt_sdpa!(o, MtlArray(Float32.(qh)), MtlArray(Float32.(kh)),
                           MtlArray(Float32.(vh)), scale) !== nothing
            o32 = MtlArray(fill(Float32(NaN), E, Lq, H, B))
            @test bt_sdpa!(o32, MtlArray(Float32.(qh)), MtlArray(Float32.(kh)),
                           MtlArray(Float32.(vh)), scale) !== nothing
            Metal.synchronize()
            @test all(isfinite, Array(o))
            @test maximum(abs, Float64.(Array(o)) .- ref) / maximum(abs, ref) < 3e-3
            @test maximum(abs, Float64.(Array(o32)) .- ref) / maximum(abs, ref) < e32
        end

        # A projection leaves q, k and v as three windows on ONE buffer. Both have to
        # come out the same: the library reads the window where it lies, by slicing
        # the dense array inside the graph, and a copy of the same bytes into a packed
        # array is what that has to agree with.
        @testset "attention reads three windows of one buffer" begin
            E, H, L, B = 16, 2, 16, 2
            scale = Float32(inv(sqrt(E)))
            # `(E, H, 3, L, B)`, which is the layout a fused qkv projection writes.
            ph = Float16.(reshape(0.3 .* sin.(range(0, 11, E*H*3*L*B)), E, H, 3, L, B))
            parent = MtlArray(ph)
            st = (1, E, E * H, E * H * 3, E * H * 3 * L)
            window(j) = (; res = parent, dims = (E, L, H, B),
                         strides = (st[1], st[4], st[2], st[5]), offset = (j - 1) * st[3])
            packed(j) = Float16.(permutedims(ph[:, :, j, :, :], (1, 3, 2, 4)))

            o  = MtlArray(fill(Float16(NaN), E, L, H, B))
            od = MtlArray(fill(Float16(NaN), E, L, H, B))
            @test bt_sdpa!(o, window(1), window(2), window(3), scale) !== nothing
            @test bt_sdpa!(od, MtlArray(packed(1)), MtlArray(packed(2)),
                           MtlArray(packed(3)), scale) !== nothing
            Metal.synchronize()
            @test all(isfinite, Array(o))
            @test Array(o) == Array(od)
            @test maximum(abs, Float64.(Array(o)) .-
                          bt_attn_ref(packed(1), packed(2), packed(3), Float64(scale))) /
                  maximum(abs, bt_attn_ref(packed(1), packed(2), packed(3), Float64(scale))) < 3e-3
        end
    end

    @testset "which strided views are windows on a dense array" begin
        parent = MtlArray(zeros(Float16, 16 * 2 * 3 * 16 * 2))
        view4(st, off) = (; res = parent, dims = (16, 16, 2, 2), strides = st, offset = off)
        # SAM 2.1's own: the middle axis of `(E, H, 3, L, B)` is the gap.
        p = BTG.packoperand(view4((1, 96, 16, 1536), 32))
        @test p !== nothing
        @test p.memdims == (16, 2, 3, 16, 2)
        @test p.tags == (1, 3, 0, 2, 4)
        @test p.starts == [1]
        # A stride that is not a multiple of what precedes it has no dense array to be
        # a window on, and an offset that does not land on a gap names no window.
        @test BTG.packoperand(view4((1, 100, 16, 1600), 0)) === nothing
        @test BTG.packoperand(view4((1, 96, 16, 1536), 33)) === nothing
        # A dense array is its own window, with no gaps and nothing to reorder.
        d = BTG.packoperand(MtlArray(zeros(Float16, 4, 5)))
        @test d.memdims == (4, 5) && isempty(d.gapat) && d.tags == (1, 2)
    end
end
