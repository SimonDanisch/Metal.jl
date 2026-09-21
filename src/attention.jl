# Fused scaled-dot-product attention on `tensor_ops::matmul2d`.
#
# The three-pass form — scores, softmax, apply — has to MATERIALISE the `Lq x Lk` score
# matrix, and at SAM 2.1's 4096-token global block that is 256 MiB written by the first
# pass, read and rewritten by the second and read again by the third. A gigabyte of traffic
# for 38.7 GFLOP of arithmetic, which is why all three passes sit within a factor of two of
# this machine's memory bandwidth and nowhere near its compute.
#
# Fused, the score tile never leaves threadgroup memory: one threadgroup owns `BQ` queries
# of one (head, batch), walks the keys in tiles of `BK`, and keeps a running max, a running
# sum and an `E x BQ` Float32 output accumulator — the standard online-softmax recurrence.
# Both products are `matmul2d`, and the second one reads its right operand out of
# THREADGROUP memory, which the run helpers support per operand.
#
# Measured on an M5, fp16, E = 72, against the three declared passes it replaces (each
# timed inside the same recorded plan):
#
#   Lq = Lk = 4096, 8 heads     22.16 -> 12.99 ms   1.71x
#   Lq = Lk =  256, 128 windows  1.44 ->  1.02 ms   1.41x
#
# `BQ = 16, BK = 128` with four simdgroups is the best of 40 tilings tried, on BOTH shapes,
# so the tiling is not a per-shape choice here. Bigger `BQ` loses to its accumulator's
# threadgroup footprint (occupancy) and smaller `BK` to the score product's aspect ratio.

"""
    attn_flash_tensor_body!(O, Q, K, V, scale, Lq, Lk, Val(E), Val(BQ), Val(BK), Val(NSIMD))

One threadgroup computes `BQ` queries of one (head, batch) plane.

The first two axes of each operand hold its matrix and every trailing axis is flattened
into the grid's y — `Q`/`O` are `(E, Lq, batch...)` and `K`/`V` are `(E, Lk, batch...)`.
"""
@inline function attn_flash_tensor_body!(O::MtlDeviceArray, Q::MtlDeviceArray,
                                        K::MtlDeviceArray, V::MtlDeviceArray,
                                        scale::Float32, Lq::UInt32, Lk::UInt32,
                                        ::Val{E}, ::Val{BQ}, ::Val{BK},
                                        ::Val{NSIMD}) where {E, BQ, BK, NSIMD}
    tgid = threadgroup_position_in_grid_3d()
    qb = (unsafe_trunc(Int32, tgid.x) - Int32(1)) * Int32(BQ)
    hb = unsafe_trunc(Int32, tgid.y) - Int32(1)

    Qp = MtlDeviceArray((E, Int(Lq)), pointer(Q, Int(hb) * E * Int(Lq) + 1))
    Kp = MtlDeviceArray((E, Int(Lk)), pointer(K, Int(hb) * E * Int(Lk) + 1))
    Vp = MtlDeviceArray((E, Int(Lk)), pointer(V, Int(hb) * E * Int(Lk) + 1))
    Op = MtlDeviceArray((E, Int(Lq)), pointer(O, Int(hb) * E * Int(Lq) + 1))
    tQ = MtlInlineTensor(Qp); tK = MtlInlineTensor(Kp); tV = MtlInlineTensor(Vp)

    # The score tile in Float32 and the probabilities in the operand type, because the
    # apply product's right operand has to match `V`'s element type.
    Sf = MtlThreadGroupArray(Float32, (BQ, BK), Val(0))
    Ph = MtlThreadGroupArray(eltype(V), (BQ, BK), Val(1))
    Oa = MtlThreadGroupArray(Float32, (E, BQ), Val(2))
    mv = MtlThreadGroupArray(Float32, (BQ,), Val(3))   # running row max
    lv = MtlThreadGroupArray(Float32, (BQ,), Val(4))   # running row sum
    cv = MtlThreadGroupArray(Float32, (BQ,), Val(5))   # this tile's rescale factor

    mSf = view(MtlInlineTensor(Sf), (Int32(1), Int32(1)), (Int32(BQ), Int32(BK)))
    mPh = view(MtlInlineTensor(Ph), (Int32(1), Int32(1)), (Int32(BQ), Int32(BK)))
    mOa = view(MtlInlineTensor(Oa), (Int32(1), Int32(1)), (Int32(E), Int32(BQ)))

    # `S = Qᵀ K` for this tile, and `O += V Pᵀ`. The contraction lengths are the tile's own
    # (E and BK), so both descriptors are fully static.
    score = TensorOpsMatmul2D{matmul2d_descriptor(BQ, BK, E; transpose_left = true,
                                                  mode = matmul2d_multiply),
                              Int32(NSIMD)}()
    apply = TensorOpsMatmul2D{matmul2d_descriptor(E, BQ, BK; transpose_right = true,
                                                  mode = matmul2d_multiply_accumulate),
                              Int32(NSIMD)}()

    nthr = NSIMD * 32
    tid = Int(thread_index_in_threadgroup()) - 1
    for lin in tid:nthr:(E * BQ - 1)
        @inbounds Oa[lin % E + 1, lin ÷ E + 1] = 0.0f0
    end
    # Strided over the rows rather than one row per thread: `nthr` may be smaller than `BQ`,
    # and a row no thread owns keeps whatever threadgroup memory held, which comes out as a
    # NaN in the answer rather than as a launch that fails.
    for r in (tid + 1):nthr:BQ
        @inbounds mv[r] = -Inf32
        @inbounds lv[r] = 0.0f0
    end
    threadgroup_barrier(MemoryFlagThreadGroup)

    # Dynamic trip count: a compile-time constant one crashes Apple's back-end (see
    # `matmul2d_descriptor`).
    ntiles = unsafe_trunc(Int32, Lk ÷ UInt32(BK))
    for t in Int32(0):(ntiles - Int32(1))
        koff = t * Int32(BK)
        score(view(tQ, (Int32(1), qb + Int32(1)), (Int32(E), Int32(BQ))),
              view(tK, (Int32(1), koff + Int32(1)), (Int32(E), Int32(BK))), mSf)
        threadgroup_barrier(MemoryFlagThreadGroup)
        # One thread per query row: the max over this tile, then the exponentials against
        # the running max. Splitting a row across threads was tried and is not faster — the
        # reduction's two extra barriers cost what the parallelism saves.
        for r in (tid + 1):nthr:BQ
            @inbounds mold = mv[r]
            mx = mold
            for j in 1:BK
                @inbounds mx = max(mx, Sf[r, j] * scale)
            end
            # `exp(-Inf - mx)` is 0, which is what the first tile needs.
            corr = exp(mold - mx)
            a = 0.0f0
            for j in 1:BK
                @inbounds e = exp(Sf[r, j] * scale - mx)
                @inbounds Ph[r, j] = eltype(V)(e)
                a += e
            end
            @inbounds mv[r] = mx
            @inbounds lv[r] = lv[r] * corr + a
            @inbounds cv[r] = corr
        end
        threadgroup_barrier(MemoryFlagThreadGroup)
        # The accumulator is rescaled to the new max BEFORE the product adds to it.
        for lin in tid:nthr:(E * BQ - 1)
            i = lin % E; j = lin ÷ E
            @inbounds Oa[i + 1, j + 1] *= cv[j + 1]
        end
        threadgroup_barrier(MemoryFlagThreadGroup)
        apply(view(tV, (Int32(1), koff + Int32(1)), (Int32(E), Int32(BK))), mPh, mOa)
        threadgroup_barrier(MemoryFlagThreadGroup)
    end

    for lin in tid:nthr:(E * BQ - 1)
        i = lin % E; j = lin ÷ E
        @inbounds Op[i + 1, Int(qb) + j + 1] = eltype(O)(Oa[i + 1, j + 1] / lv[j + 1])
    end
    return
end

function attn_flash_tensor_kernel!(O::MtlDeviceArray, Q::MtlDeviceArray,
                                   K::MtlDeviceArray, V::MtlDeviceArray,
                                   scale::Float32, Lq::UInt32, Lk::UInt32,
                                   e::Val, bq::Val, bk::Val, nsimd::Val)
    attn_flash_tensor_body!(O, Q, K, V, scale, Lq, Lk, e, bq, bk, nsimd)
end

# The measured tiling, and the fallbacks for a sequence it does not divide. `BQ = 16`,
# `BK = 128` and four simdgroups is the best of 40 combinations on both of SAM 2.1's
# attention shapes; the shorter `BK`s are there so a 64- or 32-long key run still fuses
# rather than falling back to three passes, and they were measured at 1.2-1.5x as well.
const ATTN_FLASH_BQ = (16, 8)
const ATTN_FLASH_BK = (128, 64, 32)
const ATTN_FLASH_NSIMD = 4

"""
    attention_kernel_config(O, Q, K, V; scale) -> config or nothing

Describe Metal's fused attention kernel for `O = softmax(scale · QᵀK) V`, one matrix per
trailing-axis batch entry, or `nothing` when this device or these operands are not covered.

`Q`/`O` are `(E, Lq, batch...)` and `K`/`V` are `(E, Lk, batch...)`, all with the same
element type; the answer is written in `O`'s. The head width `E` is a compile-time
parameter of the kernel — it sizes the accumulator — so a new `E` compiles a new kernel.
"""
function attention_kernel_config(O, Q, K, V; scale)
    tensor_matmul_capable() || return nothing
    T = eltype(Q)
    (eltype(K) === T && eltype(V) === T) || return nothing
    gemm_tensor_eltype(T, T, T) || return nothing
    (ndims(Q) >= 3 && ndims(K) >= 3 && ndims(V) >= 3 && ndims(O) >= 3) || return nothing

    E, Lq = size(Q, 1), size(Q, 2)
    Lk = size(K, 2)
    (size(K, 1) == E && size(V, 1) == E && size(V, 2) == Lk) || return nothing
    (size(O, 1) == E && size(O, 2) == Lq) || return nothing
    nbatch = prod(size(O)[3:end])
    (prod(size(Q)[3:end]) == nbatch && prod(size(K)[3:end]) == nbatch &&
     prod(size(V)[3:end]) == nbatch) || return nothing
    # `matmul2d` tiles in eights on every axis, and `E` is one of its contraction lengths.
    E % GEMM_TENSOR_MN_MULT == 0 || return nothing

    bq = 0
    for c in ATTN_FLASH_BQ
        Lq % c == 0 && (bq = c; break)
    end
    bk = 0
    for c in ATTN_FLASH_BK
        Lk % c == 0 && (bk = c; break)
    end
    (bq == 0 || bk == 0) && return nothing
    # Sf + Ph + Oa + the three row vectors, against what a threadgroup gets.
    tgmem = bq * bk * sizeof(Float32) + bq * bk * sizeof(T) +
            E * bq * sizeof(Float32) + 3 * bq * sizeof(Float32)
    tgmem <= GEMM_TENSOR_TGMEM || return nothing

    # 128 threads, which is under every Metal device's threadgroup limit.
    threads = ATTN_FLASH_NSIMD * 32
    # `Int` tile parameters, not `Int32`: the body builds `(E, Int(Lq))` shape tuples and a
    # mixed `Tuple{Int32, Int64}` matches no `MtlDeviceArray` constructor — which reaches
    # Mantle's access walk as "every path through it throws" rather than as a method error.
    args = (O, Q, K, V, Float32(scale), UInt32(Lq), UInt32(Lk),
            Val(Int(E)), Val(Int(bq)), Val(Int(bk)), Val(Int(ATTN_FLASH_NSIMD)))
    return (; kernel = attn_flash_tensor_kernel!, args,
            ndrange = ((Lq ÷ bq) * threads, nbatch, 1), group = (threads, 1, 1))
end
