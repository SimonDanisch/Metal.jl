# Apple's fused scaled dot-product attention, as one graph node.
#
# `scaledDotProductAttentionWithQueryTensor:` is QKᵀ → scale → softmax → ·V in one
# op, and on an M5 it is around 3.5x a hand-written fused kernel over the same
# operands (72-wide heads, 4096 keys, fp16). Worth reaching for before writing
# anything, which is what this file exists to make easy.
#
# Julia's `(E, Lq, H, B)` is MPSGraph's `(B, H, Lq, E)` — the layout Apple's SDPA
# documents — so no transposition is needed anywhere; `placeholderTensor` reverses
# the shape it is given and `MPSNDArray` describes the same bytes innermost-first.

"""What makes two SDPA graphs the same graph: the shapes, the type, and the scale."""
struct SDPAGraphKey
    dims_q::NTuple{4,Int}
    dims_k::NTuple{4,Int}
    eltyp::DataType
    scale::Float32
end

"""One built SDPA graph and the four tensors a call binds."""
struct CachedSDPAGraph
    graph::MPSGraph
    qph::MPSGraphTensor
    kph::MPSGraphTensor
    vph::MPSGraphTensor
    out::MPSGraphTensor
end

function CachedSDPAGraph(key::SDPAGraphKey)
    graph = MPSGraph()
    T = key.eltyp
    qph = placeholderTensor(graph, key.dims_q, T, "q")
    kph = placeholderTensor(graph, key.dims_k, T, "k")
    vph = placeholderTensor(graph, key.dims_k, T, "v")
    out = scaledDotProductAttentionWithQueryTensor(graph, qph, kph, vph, key.scale)
    return CachedSDPAGraph(graph, qph, kph, vph, out)
end

const _sdpa_graph_cache = Dict{SDPAGraphKey,CachedSDPAGraph}()
const _sdpa_graph_cache_lock = ReentrantLock()

"""
    sdpa_shape_supported(o, q, k, v) -> Bool

Whether Apple's SDPA covers these operands as they lie.

Rank four with the key and value runs agreeing, one element type across all four,
and an innermost extent whose byte size is a multiple of sixteen — `MPSNDArray`
refuses anything else, and refusing here is how a caller keeps its own kernel.
"""
function sdpa_shape_supported(o, q, k, v)
    # Asked of GRAPH RESOURCES as often as of arrays — a backend answers this while
    # declaring, before anything is placed — so nothing here is more specific than
    # `eltype`, `ndims` and `size`.
    all(x -> applicable(ndims, x) && applicable(eltype, x), (o, q, k, v)) || return false
    T = eltype(q)
    (T === Float16 || T === Float32) || return false
    eltype(o) === eltype(k) === eltype(v) === T || return false
    ndims(o) == ndims(q) == ndims(k) == ndims(v) == 4 || return false
    size(k) == size(v) || return false
    size(o) == size(q) || return false
    size(q, 1) == size(k, 1) || return false               # the head width
    size(q, 3) == size(k, 3) && size(q, 4) == size(k, 4) || return false
    size(q, 1) * sizeof(T) % 16 == 0 || return false
    return true
end

"""
    sdpa_batched!(o, q, k, v, scale) -> o

`o = softmax(scale · qᵀk) · v` per `(head, batch)` plane, Apple's fused op, encoded
into the current batch's command buffer.

Operands are `(E, L, H, B)` and may be suballocated; the graph is built once per
(shape, type, scale) and kept.
"""
function sdpa_batched!(o::Metal.MtlArray{T,4}, q::Metal.MtlArray{T,4},
                       k::Metal.MtlArray{T,4}, v::Metal.MtlArray{T,4},
                       scale::Real) where {T}
    key = SDPAGraphKey(size(q), size(k), T, Float32(scale))
    cached = @lock _sdpa_graph_cache_lock get!(_sdpa_graph_cache, key) do
        CachedSDPAGraph(key)
    end
    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
        cached.qph => tensordata(q),
        cached.kph => tensordata(k),
        cached.vph => tensordata(v),
    )
    results = Dict{MPSGraphTensor,MPSGraphTensorData}(cached.out => tensordata(o))
    encode_batched!(cached.graph, feeds, results, o, q, k, v)
    return o
end
