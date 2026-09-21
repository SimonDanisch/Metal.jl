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

"""What makes two `gemm_batched!` graphs the same graph."""
struct GemmGraphKey
    size_a::Tuple{Int,Int}
    size_b::Tuple{Int,Int}
    Tab::DataType
    Tc::DataType
    # `nothing` for a product with no bias, otherwise the bias element type: the
    # bias is a placeholder of its own and its cast is part of the graph.
    Tbias::Union{DataType,Nothing}
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
    placeBias = nothing
    if key.Tbias !== nothing
        placeBias = placeholderTensor(graph, (M,), key.Tbias, "bias")
        # A length-`M` bias against MPS's `(N, M)`: the trailing axis matches and
        # MPS broadcasts the leading one, which is Julia's "one value per ROW".
        b = key.Tbias === key.Tab ? placeBias :
            castTensor(graph, placeBias, key.Tab, "castbias")
        prod = additionWithPrimaryTensor(graph, prod, b)
    end
    result = key.Tc === key.Tab ? prod : castTensor(graph, prod, key.Tc, "castC")
    return CachedGemmGraph(graph, placeA, placeB, placeBias, result)
end

const _gemm_graph_cache = Dict{GemmGraphKey,CachedGemmGraph}()
const _gemm_graph_cache_lock = ReentrantLock()

"""
    gemm_batched!(C, A, B, bias = nothing) -> C

`C = A * B` (plus one value per row of `C`), encoded into the command buffer the
current queue is batching into rather than one of its own.

Operands may be suballocated — they go through `tensordata`, which carries the
offset. The graph is built once per (shape, type, bias) and kept.
"""
function gemm_batched!(C::Metal.MtlMatrix, A::Metal.MtlMatrix, B::Metal.MtlMatrix,
                       bias = nothing)
    size(A, 2) == size(B, 1) && size(C) == (size(A, 1), size(B, 2)) ||
        throw(DimensionMismatch("gemm_batched!: $(size(A)) * $(size(B)) into $(size(C))"))
    bias === nothing || length(bias) == size(C, 1) ||
        throw(DimensionMismatch("gemm_batched!: bias of $(length(bias)) for " *
                                "$(size(C, 1)) rows"))
    key = GemmGraphKey(size(A), size(B), eltype(A), eltype(C),
                       bias === nothing ? nothing : eltype(bias))
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
function gemm_shape_supported(C, A, B, bias)
    # Asked of GRAPH RESOURCES as often as of arrays — a backend answers this while
    # declaring, before anything is placed — so nothing here is more specific than
    # `eltype`, `ndims` and `size`.
    all(x -> applicable(ndims, x) && applicable(eltype, x), (C, A, B)) || return false
    ndims(C) == ndims(A) == ndims(B) == 2 || return false
    Tab = eltype(A)
    eltype(B) === Tab || return false
    (Tab, eltype(C)) in MPSGRAPH_VALID_MATMUL_TYPES || return false
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
