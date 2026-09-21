using Test, Metal

# Apple's direct 2-D convolution, in the reversed layout a declared graph uses:
# `x` is `(W, H, Cin, N)`, `w` is `(KW, KH, Cin ÷ groups, Cout)`, `out` is
# `(OW, OH, Cout, N)`. MPSGraph sees those as `NCHW` and `OIHW` with no transposition,
# which is the whole reason this is cheap to reach for — and the thing a wrong
# descriptor gets silently wrong, so every geometry below is checked against a
# reference rather than against another Metal path.

const CVG = Metal.MPSGraphs

"""The convolution in Float64, in the same layout, from the definition."""
function cv_ref(x, w, b, stride, pad, dil, groups)
    W, H, Cin, N = size(x); KW, KH, Cing, Cout = size(w)
    OW = (W + 2pad[1] - dil[1] * (KW - 1) - 1) ÷ stride[1] + 1
    OH = (H + 2pad[2] - dil[2] * (KH - 1) - 1) ÷ stride[2] + 1
    out = zeros(Float64, OW, OH, Cout, N)
    cpg = Cout ÷ groups
    for n in 1:N, co in 1:Cout
        g = (co - 1) ÷ cpg
        for oy in 1:OH, ox in 1:OW
            acc = b === nothing ? 0.0 : Float64(b[co])
            for ci in 1:Cing, ky in 1:KH, kx in 1:KW
                ix = (ox - 1) * stride[1] - pad[1] + (kx - 1) * dil[1] + 1
                iy = (oy - 1) * stride[2] - pad[2] + (ky - 1) * dil[2] + 1
                (1 <= ix <= W && 1 <= iy <= H) || continue
                acc += Float64(x[ix, iy, g * Cing + ci, n]) * Float64(w[kx, ky, ci, co])
            end
            out[ox, oy, co, n] = acc
        end
    end
    out
end

@testset "MPSGraph convolution" begin
    @testset "geometry — $KW×$KH s$(st) p$(pd) d$(dl) g$gr" for
            (W, Cin, Cout, KW, KH, st, pd, dl, gr) in
            ((32,  8, 16, 3, 3, (1, 1), (1, 1), (1, 1), 1),
             (64, 16,  8, 7, 7, (4, 4), (3, 3), (1, 1), 1),   # SAM 2.1's stem shape
             (32,  8,  8, 1, 1, (1, 1), (0, 0), (1, 1), 1),
             (32, 16, 16, 3, 3, (2, 2), (1, 1), (2, 2), 1),   # strided AND dilated
             (32, 12,  8, 3, 5, (1, 2), (1, 2), (1, 1), 1),   # asymmetric in x and y
             (16, 16,  8, 3, 3, (1, 1), (1, 1), (1, 1), 4))   # grouped
        xh = Float16.(0.2f0 .* randn(Float32, W, W, Cin, 1))
        wh = Float16.(0.2f0 .* randn(Float32, KW, KH, Cin ÷ gr, Cout))
        bh = Float16.(0.3f0 .* randn(Float32, Cout))
        want = cv_ref(xh, wh, bh, st, pd, dl, gr)
        o = MtlArray(fill(Float16(NaN), size(want, 1), size(want, 2), Cout, 1))
        @test CVG.conv2d_shape_supported(o, MtlArray(xh), MtlArray(wh), MtlArray(bh))
        CVG.conv2d_batched!(o, MtlArray(xh), MtlArray(wh), MtlArray(bh), st, pd, dl, gr)
        Metal.synchronize()
        got = Float64.(Array(o))
        # The destination starts as NaN, so an output the op did not cover cannot pass,
        # and a descriptor that got the geometry wrong lands nowhere near the reference.
        @test all(isfinite, got)
        @test maximum(abs, got .- want) / maximum(abs, want) < 2e-3
    end

    @testset "the activation and the bias on the store" begin
        xh = Float16.(0.4f0 .* randn(Float32, 24, 24, 8, 1))
        wh = Float16.(0.4f0 .* randn(Float32, 3, 3, 8, 8))
        bh = Float16.(0.5f0 .* randn(Float32, 8))
        want = cv_ref(xh, wh, bh, (1, 1), (1, 1), (1, 1), 1)
        o = MtlArray(fill(Float16(NaN), 24, 24, 8, 1))
        CVG.conv2d_batched!(o, MtlArray(xh), MtlArray(wh), MtlArray(bh),
                            (1, 1), (1, 1), (1, 1), 1, :relu)
        Metal.synchronize()
        @test maximum(abs, Float64.(Array(o)) .- max.(want, 0.0)) /
              maximum(abs, want) < 2e-3
        # No bias, and the same convolution: the bias placeholder is the only thing
        # that goes away, and it must not shift the rest.
        o2 = MtlArray(fill(Float16(NaN), 24, 24, 8, 1))
        CVG.conv2d_batched!(o2, MtlArray(xh), MtlArray(wh), nothing,
                            (1, 1), (1, 1), (1, 1), 1)
        Metal.synchronize()
        w0 = cv_ref(xh, wh, nothing, (1, 1), (1, 1), (1, 1), 1)
        @test maximum(abs, Float64.(Array(o2)) .- w0) / maximum(abs, w0) < 2e-3
    end

    # A weight's innermost extent is its KERNEL width, which is never a multiple of
    # sixteen bytes — and `MPSNDArray` over a buffer pads the innermost row to
    # sixteen, then refuses a buffer too small for the padded layout (a `(7,7,3,8)`
    # half array is 2352 bytes and it asks for 2688). So a dense operand like that is
    # bound FLAT and reshaped in the graph. This is what says the rule is understood
    # rather than worked around.
    @testset "an unaligned innermost extent is bound flat" begin
        @test CVG.bindshape((64, 64, 8, 1), Float16) === (64, 64, 8, 1)
        @test CVG.bindshape((7, 7, 3, 144), Float16) === (7 * 7 * 3 * 144,)
        @test CVG.bindshape((3, 3, 8, 8), Float16) === (3 * 3 * 8 * 8,)
        # Neither shape works: 3 halves is 6 bytes and so is the whole thing.
        @test CVG.bindshape((3, 1, 1, 1), Float16) === nothing
        # …and the convolution refuses such an operand rather than binding it wrong.
        @test !CVG.conv2d_shape_supported(MtlArray(zeros(Float16, 3, 1, 1, 1)),
                                          MtlArray(zeros(Float16, 3, 1, 1, 1)),
                                          MtlArray(zeros(Float16, 1, 1, 1, 1)), nothing)
    end

    @testset "which operands it admits" begin
        mk(T, d...) = MtlArray(zeros(T, d...))
        @test CVG.conv2d_shape_supported(mk(Float16, 8, 8, 4, 1), mk(Float16, 8, 8, 4, 1),
                                         mk(Float16, 3, 3, 4, 4), nothing)
        # A destination wider than the operands, for the reason the product refuses
        # one: there is no compute type to ask for.
        @test !CVG.conv2d_shape_supported(mk(Float32, 8, 8, 4, 1), mk(Float16, 8, 8, 4, 1),
                                          mk(Float16, 3, 3, 4, 4), nothing)
        # Rank, a channel count that does not match the weight, an activation with no
        # node, and a bias that is not one value per output channel.
        @test !CVG.conv2d_shape_supported(mk(Float16, 8, 8, 4), mk(Float16, 8, 8, 4),
                                          mk(Float16, 3, 3, 4, 4), nothing)
        @test !CVG.conv2d_shape_supported(mk(Float16, 8, 8, 8, 1), mk(Float16, 8, 8, 4, 1),
                                          mk(Float16, 3, 3, 4, 4), nothing)
        @test !CVG.conv2d_shape_supported(mk(Float16, 8, 8, 4, 1), mk(Float16, 8, 8, 4, 1),
                                          mk(Float16, 3, 3, 4, 4), nothing, :softplus)
        @test !CVG.conv2d_shape_supported(mk(Float16, 8, 8, 4, 1), mk(Float16, 8, 8, 4, 1),
                                          mk(Float16, 3, 3, 4, 4), mk(Float16, 8))
    end

    # SAM 2.1's stem at its real size, against the SAME graph in single. This is the
    # measurement that settles whether the library convolution costs accuracy: the
    # answer is half's own output rounding and not one bit more, so `add_129` moving
    # when the stem changes route is a different valid rounding rather than a loss.
    @testset "the stem is at half's rounding floor" begin
        Cout = 144
        xh = Float16.(0.3f0 .* randn(Float32, 1024, 1024, 3, 1))
        wh = Float16.(0.05f0 .* randn(Float32, 7, 7, 3, Cout))
        bh = Float16.(0.2f0 .* randn(Float32, Cout))
        o16 = MtlArray(fill(Float16(NaN), 256, 256, Cout, 1))
        o32 = MtlArray(fill(NaN32, 256, 256, Cout, 1))
        CVG.conv2d_batched!(o16, MtlArray(xh), MtlArray(wh), MtlArray(bh),
                            (4, 4), (3, 3), (1, 1), 1)
        CVG.conv2d_batched!(o32, MtlArray(Float32.(xh)), MtlArray(Float32.(wh)),
                            MtlArray(Float32.(bh)), (4, 4), (3, 3), (1, 1), 1)
        Metal.synchronize()
        a, b = Float64.(Array(o16)), Float64.(Array(o32))
        sc = maximum(abs, b)
        @test all(isfinite, a)
        # Half's own spacing at that magnitude, with a hair of slack. An accumulator
        # that was half rather than single would be an order of magnitude over this.
        @test maximum(abs, a .- b) / sc < 1.2 * Float64(eps(Float16(sc))) / 2 / sc
    end
end
