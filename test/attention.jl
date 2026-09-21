using Test, Metal

# Fused attention: the arithmetic, the online-softmax recurrence, and which operands the
# config admits. `find_tests` picks this file up by being here.

"""`softmax(scale·QᵀK)·V` per (head, batch) plane, in Float64 over the operands given."""
function attn_ref(qh, kh, vh, scale)
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

"""Run the fused kernel through its own config, the way a graph runtime would."""
function fused_attention(qh, kh, vh, scale)
    q = MtlArray(qh); k = MtlArray(kh); v = MtlArray(vh)
    o = MtlArray(fill(eltype(qh)(NaN), size(qh)))
    cfg = Metal.attention_kernel_config(o, q, k, v; scale)
    cfg === nothing && return nothing
    groups = (cfg.ndrange[1] ÷ cfg.group[1], cfg.ndrange[2], cfg.ndrange[3])
    Metal.@metal threads=cfg.group groups=groups cfg.kernel(cfg.args...)
    Metal.synchronize()
    Array(o)
end

@testset "fused attention" begin
    if !Metal.tensor_matmul_capable()
        @test_skip Metal.tensor_matmul_capable()
    else
        E = 72                       # SAM 2.1's head width, and not a power of two
        scale = Float32(inv(sqrt(E)))

        # Two shapes on purpose: 128 keys is ONE tile, so the running max and sum are
        # written once and never corrected, and 256 is two, which is the only way the
        # rescale path runs at all.
        @testset "matches a Float64 reference — Lq=$Lq Lk=$Lk" for (Lq, Lk, H, B) in
                ((128, 128, 2, 1), (256, 256, 2, 2), (16, 128, 3, 1))
            qh = Float16.(reshape(0.4 .* sin.(range(0, 9, E*Lq*H*B)), E, Lq, H, B))
            kh = Float16.(reshape(0.4 .* cos.(range(0, 7, E*Lk*H*B)), E, Lk, H, B))
            vh = Float16.(reshape(0.4 .* sin.(range(0, 5, E*Lk*H*B)), E, Lk, H, B))
            got = fused_attention(qh, kh, vh, scale)
            @test got !== nothing
            ref = attn_ref(qh, kh, vh, Float64(scale))
            # A kernel that writes nothing also "matches" a zero reference, and the
            # destination starts as NaN so a tile it skips cannot pass either.
            @test all(isfinite, got)
            @test maximum(abs, Float64.(got) .- ref) / maximum(abs, ref) < 3e-3
        end

        # The RUNNING MAX, which a small random test cannot see. Scores here reach ~100, so
        # exponentiating them without subtracting a max gives `Inf` and then `NaN/NaN`; and
        # the largest score is in the LAST key tile, so the accumulator has to be rescaled
        # after it has already been written.
        @testset "the online softmax subtracts a running max" begin
            Lq, Lk, H, B = 32, 256, 1, 1
            qh = fill(Float16(2), E, Lq, H, B)
            kh = Float16.(reshape(repeat(range(0.5f0, 6.0f0; length = Lk)', E), E, Lk, H, B))
            vh = Float16.(reshape(0.3 .* sin.(range(0, 5, E*Lk*H*B)), E, Lk, H, B))
            raw = maximum(Float64(scale) * sum(Float64(qh[e,1,1,1]) * Float64(kh[e,lk,1,1])
                                               for e in 1:E) for lk in 1:Lk)
            @test raw > 90            # `exp` of this overflows Float32
            got = fused_attention(qh, kh, vh, scale)
            @test got !== nothing
            @test all(isfinite, got)
            ref = attn_ref(qh, kh, vh, Float64(scale))
            @test maximum(abs, Float64.(got) .- ref) / maximum(abs, ref) < 3e-3
        end

        @testset "which operands the config admits" begin
            mk(T, dims) = MtlArray(fill(T(0), dims...))
            ok = Metal.attention_kernel_config(mk(Float16, (E, 256, 2, 1)),
                                               mk(Float16, (E, 256, 2, 1)),
                                               mk(Float16, (E, 256, 2, 1)),
                                               mk(Float16, (E, 256, 2, 1)); scale = 0.1f0)
            @test ok !== nothing
            @test ok.group == (Metal.ATTN_FLASH_NSIMD * 32, 1, 1)
            # A key run no tile divides, and a query run no tile divides.
            @test Metal.attention_kernel_config(mk(Float16, (E, 256, 2, 1)),
                                                mk(Float16, (E, 256, 2, 1)),
                                                mk(Float16, (E, 48, 2, 1)),
                                                mk(Float16, (E, 48, 2, 1)); scale = 0.1f0) === nothing
            @test Metal.attention_kernel_config(mk(Float16, (E, 12, 2, 1)),
                                                mk(Float16, (E, 12, 2, 1)),
                                                mk(Float16, (E, 128, 2, 1)),
                                                mk(Float16, (E, 128, 2, 1)); scale = 0.1f0) === nothing
            # A head width `matmul2d` cannot tile, mixed element types, and a plane with no
            # batch axis at all.
            @test Metal.attention_kernel_config(mk(Float16, (70, 128, 2, 1)),
                                                mk(Float16, (70, 128, 2, 1)),
                                                mk(Float16, (70, 128, 2, 1)),
                                                mk(Float16, (70, 128, 2, 1)); scale = 0.1f0) === nothing
            @test Metal.attention_kernel_config(mk(Float16, (E, 128, 2, 1)),
                                                mk(Float16, (E, 128, 2, 1)),
                                                mk(Float32, (E, 128, 2, 1)),
                                                mk(Float16, (E, 128, 2, 1)); scale = 0.1f0) === nothing
            @test Metal.attention_kernel_config(mk(Float16, (E, 128)), mk(Float16, (E, 128)),
                                                mk(Float16, (E, 128)),
                                                mk(Float16, (E, 128)); scale = 0.1f0) === nothing
            # Batch counts that disagree between the operands.
            @test Metal.attention_kernel_config(mk(Float16, (E, 128, 2, 1)),
                                                mk(Float16, (E, 128, 2, 1)),
                                                mk(Float16, (E, 128, 3, 1)),
                                                mk(Float16, (E, 128, 3, 1)); scale = 0.1f0) === nothing
        end
    end
end
