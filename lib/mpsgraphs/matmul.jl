#=
Creates a default MPSGraphExecutionDescriptor with a MPSGraphCompilationDescriptor
 set to use optimization level 0 instead of 1. This is because level 1 causes operations
 on eltypes <= 16 bytes to be executed on the ANE instead of the GPU, leading to worse
 performance and hangs when the matrices are too big
=#
function default_exec_desc()
    @memoize begin
        compDesc = MPSGraphCompilationDescriptor()
        # Use optimization level 0 to avoid operations being moved to the neural engine
        compDesc.optimizationLevel = MPSGraphOptimizationLevel0

        execDesc = MPSGraphExecutionDescriptor()
        execDesc.compilationDescriptor = compDesc
        execDesc
    end::MPSGraphExecutionDescriptor
end


#=
MPSGraph caching infrastructure.

The overhead of creating an MPSGraph dominates matmul time for small-medium matrices.
By caching graphs keyed by their structural parameters (shapes, types, flags), we
achieve significant speedup for repeated operations with the same configuration.

The cache key includes all parameters that affect graph structure:
- Input/output shapes and element types
- Transpose flags
- Alpha/beta values (baked into graph as constants)
=#

# Cache key for matmul graphs - includes all structural parameters
struct MatmulGraphKey{Tab<: Number, Tc <: Number}
    size_a::Tuple{Vararg{Int}}
    size_b::Tuple{Vararg{Int}}
    size_c::Tuple{Vararg{Int}}
    ndims_a::Int
    ndims_b::Int
    alpha::Tab
    beta::Tc
    transpose_a::Char
    transpose_b::Char
end
# Build graph key from matmul parameters
function MatmulGraphKey(a::MtlArray{Tab, Na}, b::MtlArray{Tab, Nb}, c::MtlArray{Tc},
                          alpha::Number, beta::Number,
                          transpose_a, transpose_b) where {Tc, Tab, Na, Nb}
    MatmulGraphKey{Tab, Tc}(
        size(a), size(b), size(c),
        Na, Nb,
        Tab(alpha), Tc(beta),
        transpose_a, transpose_b
    )
end

# Cached graph with all tensors needed for execution
struct CachedMatmulGraph
    graph::MPSGraph
    place_c::MPSGraphTensor
    place_a::MPSGraphTensor
    place_b::MPSGraphTensor
    result::MPSGraphTensor
end
# Build a new matmul graph (called only on cache miss)
function CachedMatmulGraph(key::MatmulGraphKey{Tab, Tc}) where {Tab, Tc}
    graph = MPSGraph()

    placeA = placeholderTensor(graph, key.size_a, Tab)
    placeB = placeholderTensor(graph, key.size_b, Tab)
    placeC = placeholderTensor(graph, key.size_c, Tc)

    # cast to output eltype if input type is an integer type
    castTab = Tab <: Integer ? Tc : Tab
    castA = castTensor(graph, placeA, castTab, "castA")
    castB = castTensor(graph, placeB, castTab, "castB")

    conjA = if key.transpose_a == 'C'
        conjugateWithTensor(graph, castA, "conjA")
    else
        castA
    end

    conjB = if key.transpose_b == 'C'
        conjugateWithTensor(graph, castB, "conjB")
    else
        castB
    end

    transA = (key.transpose_a == 'T' || key.transpose_a == 'C') ? transposeTensor(graph, conjA, key.ndims_a - 2, key.ndims_a - 1, "transpose_a") : conjA
    transB = (key.transpose_b == 'T' || key.transpose_b == 'C') ? transposeTensor(graph, conjB, key.ndims_b - 2, key.ndims_b - 1, "transpose_b") : conjB

    nBatchA = key.ndims_a == 2 ? 1 : key.size_a[1]
    nBatchB = key.ndims_b == 2 ? 1 : key.size_b[1]

    # for batched matmul between different sized tensors
    broadcastA, broadcastB = if nBatchA == nBatchB
        transA, transB
    elseif key.ndims_a == 1
        broadcastTensor(graph, transA, convert(MPSShape, [nBatchB, size(transA)[2:end]...])), transB
    elseif key.ndims_b == 1
        transA, broadcastTensor(graph, transB, convert(MPSShape, [nBatchA, size(transB)[2:end]...]))
    else
        transA, transB
    end

    matmul = matrixMultiplicationWithPrimaryTensor(graph, broadcastB, broadcastA)

    afteralpha = let
        alphatensor = if castTab <: Real
            constantWithScalar(graph, key.alpha, castTab)
        else
            complexConstant(graph, key.alpha, castTab)
        end
        multiplicationWithPrimaryTensor(graph, alphatensor, matmul)
    end

    castC = castTensor(graph, afteralpha, Tc, "castC")

    afterbeta = let
        betatensor = if Tc <: Real
            constantWithScalar(graph, key.beta, Tc)
        else
            complexConstant(graph, key.beta, Tc)
        end
        castplaceC = castTensor(graph, placeC, Tc, "castplaceC")
        betaC = multiplicationWithPrimaryTensor(graph, betatensor, castplaceC)
        additionWithPrimaryTensor(graph, castC, betaC)
    end

    CachedMatmulGraph(graph, placeC, placeA, placeB, afterbeta)
end

# Thread-safe graph cache with lock
const _matmul_graph_cache = Dict{MatmulGraphKey, CachedMatmulGraph}()
const _matmul_graph_cache_lock = ReentrantLock()
@autoreleasepool function _matmul!(c::MtlArray{Tc}, a::MtlArray{Tab, Na}, b::MtlArray{Tab, Nb},
                                   alpha::Number, beta::Number,
                                   transpose_a, transpose_b) where {Tc, Tab, Na, Nb}
    # Get or create cached graph
    key = MatmulGraphKey(a, b, c, alpha, beta, transpose_a, transpose_b)
    cached = @lock _matmul_graph_cache_lock get!(_matmul_graph_cache, key) do
        CachedMatmulGraph(key)
    end

    # Build feed and result dictionaries with current data
    feeds = Dict{MPSGraphTensor, MPSGraphTensorData}(
        cached.place_a => tensordata(a),
        cached.place_b => tensordata(b),
        cached.place_c => tensordata(c)
    )

    resultdict = Dict{MPSGraphTensor, MPSGraphTensorData}(
        cached.result => feeds[cached.place_c]
    )

    # Into the OPEN batch rather than a command buffer of its own. Committing one
    # here put this product ahead of every kernel launch still sitting in the batch,
    # because command buffers run in commit order — so `mul!` between two launches
    # was ordered against neither. See `encode_batched!`.
    encode_batched!(cached.graph, feeds, resultdict, a, b, c)

    return c
end

function graph_matmul!(c::MtlArray{Tc, N}, a::MtlArray{Tab, N}, b::MtlArray{Tab, N}, alpha::Number = true, beta::Number = false, transpose_a = 'N', transpose_b = 'N') where {Tc, Tab, N}
    _matmul!(c, a, b, alpha, beta, transpose_a, transpose_b)
end

function graph_matvecmul!(c::MtlVector{Tc}, a::MtlMatrix{Tab}, b::MtlVector{Tab}, alpha::Number = true, beta::Number = false, transpose = 'N') where {Tc, Tab}
    _matmul!(c, a, b, alpha, beta, transpose, 'N')
end

# ── The declared product: A*B (+ bias), batched into the caller's command buffer ──
#
# `_matmul!` above is the `mul!` entry point and commits a command buffer of its
# own. That is the wrong shape for a render graph, which has already decided where
# its submission boundaries are, and it cannot fold a bias. Both are the same graph
# with a different epilogue and a different place to put it.

"""Which activations this graph can put on a product's store."""
const GEMM_ACTIVATIONS = (:identity, :relu, :gelu, :gelu_tanh)

"""
    activation(graph, t, kind) -> MPSGraphTensor

One named activation as graph nodes, evaluated in `t`'s type.

`:gelu` is the erf formulation — torch's default — and `:gelu_tanh` its `approximate
= "tanh"` variant. Apple's `erf` is a real one, so this is not bit-identical to a
hand-written Abramowitz-Stegun approximation of the same expression; it is closer to
the reference the models are checked against, which is the comparison that matters.
"""
function activation(graph::MPSGraph, t::MPSGraphTensor, kind::Symbol, T::DataType)
    kind === :identity && return t
    kind === :relu && return reLUWithTensor(graph, t)
    half(v) = constantWithScalar(graph, v, T)
    if kind === :gelu
        # 0.5x (1 + erf(x / sqrt 2))
        inner = erfWithTensor(graph, multiplicationWithPrimaryTensor(
            graph, t, half(0.7071067811865476), "gelu_scale"), "gelu_erf")
        s = additionWithPrimaryTensor(graph, inner, half(1.0), "gelu_one")
        return multiplicationWithPrimaryTensor(
            graph, multiplicationWithPrimaryTensor(graph, t, half(0.5), "gelu_halfx"),
            s, "gelu")
    end
    if kind === :gelu_tanh
        # 0.5x (1 + tanh(sqrt(2/pi) (x + 0.044715 x³)))
        x2 = multiplicationWithPrimaryTensor(graph, t, t, "gelu_x2")
        x3 = multiplicationWithPrimaryTensor(graph, x2, t, "gelu_x3")
        inner = additionWithPrimaryTensor(graph, t, multiplicationWithPrimaryTensor(
            graph, x3, half(0.044715), "gelu_c3"), "gelu_inner")
        th = tanhWithTensor(graph, multiplicationWithPrimaryTensor(
            graph, inner, half(0.7978845608028654), "gelu_sqrt2pi"), "gelu_tanh")
        s = additionWithPrimaryTensor(graph, th, half(1.0), "gelu_one")
        return multiplicationWithPrimaryTensor(
            graph, multiplicationWithPrimaryTensor(graph, t, half(0.5), "gelu_halfx"),
            s, "gelu")
    end
    throw(ArgumentError("gemm_batched!: no node for activation :$kind"))
end

"""What makes two `gemm_batched!` graphs the same graph."""
struct GemmGraphKey
    size_a::Tuple{Int,Int}
    size_b::Tuple{Int,Int}
    Tab::DataType
    Tc::DataType
    # `nothing` for a product with no bias, otherwise the bias element type: the
    # bias is a placeholder of its own and its cast is part of the graph.
    Tbias::Union{DataType,Nothing}
    act::Symbol
end

"""One built product graph and the tensors a call binds."""
struct CachedGemmGraph
    graph::MPSGraph
    place_a::MPSGraphTensor
    place_b::MPSGraphTensor
    place_bias::Union{MPSGraphTensor,Nothing}
    result::MPSGraphTensor
end

function CachedGemmGraph(key::GemmGraphKey)
    graph = MPSGraph()
    M, K = key.size_a
    _, N = key.size_b
    placeA = placeholderTensor(graph, key.size_a, key.Tab, "A")
    placeB = placeholderTensor(graph, key.size_b, key.Tab, "B")
    # Julia's `A * B` is MPSGraph's `B * A`: the shapes are reversed, so MPS sees
    # `(N, K) * (K, M) = (N, M)`, which is Julia's `(M, N)`. Same swap `_matmul!`
    # makes, for the same reason.
    prod = matrixMultiplicationWithPrimaryTensor(graph, placeB, placeA)
    # The EPILOGUE in Float32 where there is an activation, and in the operand type
    # where there is not. `gelu` of a value rounded to half loses accuracy twice —
    # inside `erf`, and again at the `1 + erf` that follows, where an argument below
    # -2 cancels to a couple of significant bits — which is why torch evaluates it in
    # `opmath_type` and why the kernel this replaces accumulates in Float32 and
    # rounds once. Not applied to a plain product, whose structure is measured.
    wide = key.act !== :identity && key.Tab !== Float32
    epi = wide ? castTensor(graph, prod, Float32, "wide") : prod
    Te = wide ? Float32 : key.Tab
    placeBias = nothing
    if key.Tbias !== nothing
        placeBias = placeholderTensor(graph, (M,), key.Tbias, "bias")
        # A length-`M` bias against MPS's `(N, M)`: the trailing axis matches and
        # MPS broadcasts the leading one, which is Julia's "one value per ROW".
        b = key.Tbias === Te ? placeBias : castTensor(graph, placeBias, Te, "castbias")
        epi = additionWithPrimaryTensor(graph, epi, b)
    end
    epi = activation(graph, epi, key.act, Te)
    result = key.Tc === Te ? epi : castTensor(graph, epi, key.Tc, "castC")
    return CachedGemmGraph(graph, placeA, placeB, placeBias, result)
end

const _gemm_graph_cache = Dict{GemmGraphKey,CachedGemmGraph}()
const _gemm_graph_cache_lock = ReentrantLock()

"""
    gemm_batched!(C, A, B, bias = nothing) -> C

`C = act.(A * B .+ bias)`, encoded into the command buffer the current queue is
batching into rather than one of its own. `act` is one of `GEMM_ACTIVATIONS`.

Operands may be suballocated — they go through `tensordata`, which carries the
offset. The graph is built once per (shape, type, bias) and kept.
"""
function gemm_batched!(C::Metal.MtlMatrix, A::Metal.MtlMatrix, B::Metal.MtlMatrix,
                       bias = nothing, act::Symbol = :identity)
    size(A, 2) == size(B, 1) && size(C) == (size(A, 1), size(B, 2)) ||
        throw(DimensionMismatch("gemm_batched!: $(size(A)) * $(size(B)) into $(size(C))"))
    # Shapes were the only thing checked here, and the types are what ABORT.
    # This graph feeds its placeholders straight into `mps.matmul` -- unlike
    # `graph_matmul!`, which casts both operands to a common type first -- so an
    # integer operand reaches a node that has none: "operand #0 must be tensor of
    # floating point values". MPSGraph answers that by failing its own module
    # verification and calling `abort()`, which takes the Julia process with it,
    # with no exception to catch and nothing in the backtrace above Apple's
    # assert. `conv2d_batched!` guards the same way for the same reason.
    gemm_shape_supported(C, A, B, bias, act) || throw(ArgumentError(
        "gemm_batched!: these operands are not ones MPSGraph can be given — " *
        "$(size(A)) * $(size(B)) into $(size(C)), $(eltype(A))/$(eltype(B))/" *
        "$(eltype(C)), activation :$act. Ask `gemm_shape_supported` first."))
    bias === nothing || length(bias) == size(C, 1) ||
        throw(DimensionMismatch("gemm_batched!: bias of $(length(bias)) for " *
                                "$(size(C, 1)) rows"))
    key = GemmGraphKey(size(A), size(B), eltype(A), eltype(C),
                       bias === nothing ? nothing : eltype(bias), act)
    cached = @lock _gemm_graph_cache_lock get!(_gemm_graph_cache, key) do
        CachedGemmGraph(key)
    end
    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
        cached.place_a => tensordata(A),
        cached.place_b => tensordata(B),
    )
    bias === nothing || (feeds[cached.place_bias] = tensordata(bias))
    results = Dict{MPSGraphTensor,MPSGraphTensorData}(cached.result => tensordata(C))
    if bias === nothing
        encode_batched!(cached.graph, feeds, results, C, A, B)
    else
        encode_batched!(cached.graph, feeds, results, C, A, B, bias)
    end
    return C
end

"""
    gemm_shape_supported(C, A, B, bias) -> Bool

Whether `gemm_batched!` covers these operands as they lie. Rank two throughout, one
input element type, and an innermost extent whose byte size is a multiple of sixteen
— `MPSNDArray` refuses anything else.
"""
function gemm_shape_supported(C, A, B, bias, act::Symbol = :identity)
    act in GEMM_ACTIVATIONS || return false
    # Asked of GRAPH RESOURCES as often as of arrays — a backend answers this while
    # declaring, before anything is placed — so nothing here is more specific than
    # `eltype`, `ndims` and `size`.
    all(x -> applicable(ndims, x) && applicable(eltype, x), (C, A, B)) || return false
    ndims(C) == ndims(A) == ndims(B) == 2 || return false
    Tab = eltype(A)
    eltype(B) === Tab || return false
    (Tab, eltype(C)) in MPSGRAPH_VALID_MATMUL_TYPES || return false
    # A destination WIDER than the operands is refused, and this is the one that
    # matters rather than a tidiness rule. `matrixMultiplicationWithPrimaryTensor`
    # has no compute type: its output dtype follows its inputs, so a half product
    # asked for in single is a half product cast afterwards, and the accumulator
    # the caller wanted is gone. A convolution's im2col GEMM asks for exactly that —
    # `(Float16, Float16) -> Float32`, on the grounds that the native kernel
    # accumulates directly into single — and taking it here moved SAM 2.1's encoder
    # `add_129` from 0.48 to 1.12, past the band the Vulkan reference sits in. The
    # only way to give MPSGraph a wide accumulator is to widen its operands, which
    # is four times the traffic for a product the caller already has a kernel for.
    eltype(C) === Tab || return false
    size(A, 2) == size(B, 1) && size(C) == (size(A, 1), size(B, 2)) || return false
    for x in (C, A, B)
        size(x, 1) * sizeof(eltype(x)) % 16 == 0 || return false
    end
    if bias !== nothing
        applicable(ndims, bias) && applicable(eltype, bias) || return false
        ndims(bias) == 1 || return false
        length(bias) == size(C, 1) || return false
        # A vector's own innermost extent is its length, and MPS wants that
        # multiple of sixteen bytes too.
        length(bias) * sizeof(eltype(bias)) % 16 == 0 || return false
    end
    return true
end
