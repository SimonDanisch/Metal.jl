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

# ── Reading an operand where it lies ─────────────────────────────────────────
#
# A projection leaves q, k and v interleaved in one buffer — SAM 2.1's are three
# windows on the middle axis of one `(E, H, 3, L, B)` array — so the operand
# attention wants is a strided view and not an array. Materialising it costs a full
# copy of every operand of every attention op, which is most of what the library
# saves.
#
# It does not have to be copied. Such a view is a DENSE array with axes the window
# does not span, so the array goes to MPSGraph whole and the narrowing happens in
# the graph: one `sliceTensor` per gap, a reshape to drop them, and the transposes
# that put the axes back in the caller's order. All of it is graph structure, built
# once per layout and cached with the graph.

"""
An operand that is a window on a dense array: which array, how it lies in memory,
and where the window starts.

`memdims` is the dense shape innermost-first, `tags` says which of the caller's axes
each memory axis is (`0` for a GAP the window does not span), `gapat` indexes the
gaps within `memdims`, and `starts` is where the window begins along each of them.
"""
struct PackedOperand{R}
    res::R
    dims::Tuple                 # the shape the caller asked for
    memdims::Tuple
    tags::Tuple
    gapat::Vector{Int}
    starts::Vector{Int}
end

"""Everything about an operand except its bytes, which is what a graph is built on."""
packedkey(p::PackedOperand) =
    (p.dims, p.memdims, p.tags, Tuple(p.gapat), Tuple(p.starts))

"""
    packoperand(x) -> PackedOperand or nothing

One operand, dense or strided, as the dense array it lives in.

`x` is either an array — the whole of it — or a `(; res, dims, strides, offset)` view
of one, which is the shape a strided operand arrives in (`offset` in elements).
`nothing` when the view is not a window on a dense array: a stride that is not a
multiple of what precedes it has no dense array to be a window on, and neither has
an offset that does not land on a gap.

Answered for a graph RESOURCE as readily as for an array — a backend asks this while
declaring, before anything is placed — so nothing here is more specific than
`eltype`, `ndims`, `size` and `length`.
"""
function packoperand(x::NamedTuple)
    dims = Tuple(Int.(x.dims)); strides = Tuple(Int.(x.strides)); offset = Int(x.offset)
    length(dims) == length(strides) || return nothing
    # An axis of one extent occupies nothing and has no meaningful stride, so it takes
    # no part in the layout; the reshape at the end puts it back.
    real = [i for i in eachindex(dims) if dims[i] != 1]
    isempty(real) && return nothing
    order = sort(real; by = i -> strides[i])
    memdims = Int[]; tags = Int[]; gapat = Int[]; gapstride = Int[]
    acc = 1
    for p in order
        s = strides[p]
        if s != acc
            (s > acc && s % acc == 0) || return nothing
            push!(memdims, s ÷ acc); push!(tags, 0)
            push!(gapat, length(memdims)); push!(gapstride, acc)
            acc = s
        end
        push!(memdims, dims[p]); push!(tags, p)
        acc *= dims[p]
    end
    # Where the window starts, one index per gap. Largest stride first, so the
    # decomposition is the only one there is.
    starts = zeros(Int, length(gapat))
    rest = offset
    for gi in length(gapat):-1:1
        idx, rest = divrem(rest, gapstride[gi])
        idx < memdims[gapat[gi]] || return nothing
        starts[gi] = idx
    end
    rest == 0 || return nothing
    # The dense array the window is on has to fit in the bytes the resource owns.
    prod(memdims) <= length(x.res) || return nothing
    return PackedOperand(x.res, dims, Tuple(memdims), Tuple(tags), gapat, starts)
end

function packoperand(x)
    (applicable(size, x) && applicable(eltype, x) && applicable(ndims, x)) ||
        return nothing
    ndims(x) >= 1 || return nothing
    d = size(x)
    return PackedOperand(x, d, d, ntuple(identity, length(d)), Int[], Int[])
end

"""Whether MPS can bind this operand's bytes: the rank, and the sixteen-byte row."""
function packedbindable(p::PackedOperand)
    1 <= length(p.memdims) <= 16 || return false
    # `MPSNDArray` refuses an innermost extent that is not a multiple of sixteen
    # bytes, and a suballocated operand has to go through one.
    first(p.memdims) * sizeof(eltype(p.res)) % 16 == 0 || return false
    return true
end

"""
Narrow and reorder a packed operand's placeholder into the tensor the caller asked
for.

Built from the KEY alone — no array — because it is graph structure and is cached
with the graph. A dense operand costs no node at all.
"""
function packedplaceholder(graph::MPSGraph, key, t::MPSGraphTensor, name)
    dims, memdims, tags, gapat, starts = key
    M = length(memdims)
    # One index along each gap. The axis is KEPT, with extent one, so the axis
    # numbers of the slices after it do not move.
    for (gi, at) in enumerate(gapat)
        t = sliceTensor(graph, t, mps_axis(M, at), starts[gi], 1, "$(name)_s$gi")
    end
    isempty(gapat) || (t = reshapeTensor(graph, t,
        convert(MPSShape, reverse([memdims[i] for i in 1:M if tags[i] != 0])),
        "$(name)_drop"))
    # What is left is the caller's axes, in MEMORY order. Selection sort into the
    # caller's order, one transpose per swap.
    ord = [tags[i] for i in 1:M if tags[i] != 0]
    R = length(ord)
    for i in 1:R
        j = i
        for l in (i + 1):R
            ord[l] < ord[j] && (j = l)
        end
        j == i && continue
        t = transposeTensor(graph, t, mps_axis(R, i), mps_axis(R, j), "$(name)_t$i")
        ord[i], ord[j] = ord[j], ord[i]
    end
    # …and the axes of one extent, which took no part in any of the above.
    R == length(dims) || (t = reshapeTensor(graph, t, convert(MPSShape, reverse(dims)),
                                            "$(name)_fill"))
    return t
end

"""The bytes behind a packed operand, as the dense array the graph was built for."""
function packeddata(arr::MtlArray, memdims)
    memdims == size(arr) && arr.offset == 0 && return MPSGraphTensorData(arr)
    desc = MPS.MPSNDArrayDescriptor(eltype(arr), collect(memdims))
    return MPSGraphTensorData(MPS.MPSNDArray(arr.data[], UInt(arr.offset), desc))
end

# ── The op ───────────────────────────────────────────────────────────────────

"""What makes two SDPA graphs the same graph: the operand layouts, the two types and
the scale. No array, and no address.

`eltyp` is what q, k and v are; `outtyp` is what the caller asked the answer to be.
They differ where a graph projects in fp16 and declares an fp32 result, which is what
Qwen-Image 2.1 exports.

Worth being exact about what the wider destination is and is not, because the
matmul gate refuses the same pair for a reason that sounds like it applies here.
`scaledDotProductAttentionWithQueryTensor` has no compute type either: its result
dtype follows its INPUTS, so fp16 operands give an fp16-accurate answer and the cast
below widens it rather than accumulating in single. Measured against a Float64
reference at `(128, 256/384, 4, 1)`: fp16 operands read 7.06e-4 into an fp32
destination and 7.06e-4 into an fp16 one — the same number — where fp32 operands
read 9.08e-7.

It is admitted anyway, which `gemm_shape_supported`'s pair is not, because of what
the caller does when it is refused. A refused product falls back to a kernel that
accumulates in single, so refusing keeps an accumulator. A refused ATTENTION falls
back to `DNNKernels`' `threepass!`, whose score matrix is `eltype(q)` — fp16, the
same precision this gives — so refusing keeps nothing and costs the whole op.
"""
struct SDPAGraphKey
    q::Any
    k::Any
    v::Any
    eltyp::DataType
    outtyp::DataType
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
    qph = placeholderTensor(graph, key.q[2], T, "q")
    kph = placeholderTensor(graph, key.k[2], T, "k")
    vph = placeholderTensor(graph, key.v[2], T, "v")
    out = scaledDotProductAttentionWithQueryTensor(
        graph,
        packedplaceholder(graph, key.q, qph, "q"),
        packedplaceholder(graph, key.k, kph, "k"),
        packedplaceholder(graph, key.v, vph, "v"),
        key.scale)
    key.outtyp === T || (out = castTensor(graph, out, key.outtyp, "castout"))
    return CachedSDPAGraph(graph, qph, kph, vph, out)
end

const _sdpa_graph_cache = Dict{SDPAGraphKey,CachedSDPAGraph}()
const _sdpa_graph_cache_lock = ReentrantLock()

"""
    sdpa_operands(o, q, k, v) -> (; keys, res) or nothing

Whether Apple's SDPA covers these operands as they lie, and if so everything about
them a graph is built from: one layout key and one array per operand.

Rank four throughout with the key and value runs agreeing, one element type across
q, k and v, a result that is fp16 or fp32 and need not match them, a destination that
is an array rather than a view, and operands MPS can bind. `nothing` is how a caller
learns to keep its own kernel.
"""
function sdpa_operands(o, q, k, v)
    po = packoperand(o)
    po === nothing && return nothing
    # The ANSWER is written, so it has to be an array and not a window on one.
    isempty(po.gapat) && po.dims == po.memdims || return nothing
    ps = map(packoperand, (q, k, v))
    any(isnothing, ps) && return nothing
    all(packedbindable, ps) || return nothing
    # The OPERAND type and the RESULT type are asked separately. They agree in most
    # graphs and the cast costs nothing there; where they do not, requiring them to
    # agree is what sent an fp16 attention with an fp32 result to the three-pass
    # path. The wider destination does NOT buy a wider accumulator — see
    # `SDPAGraphKey` for the measurement, and for why that is worth taking here and
    # not in `gemm_shape_supported`.
    Te = eltype(first(ps).res)
    To = eltype(o)
    (Te === Float16 || Te === Float32) || return nothing
    (To === Float16 || To === Float32) || return nothing
    all(p -> eltype(p.res) === Te, ps) || return nothing
    pq, pk, pv = ps
    length(po.dims) == length(pq.dims) == length(pk.dims) == length(pv.dims) == 4 ||
        return nothing
    pk.dims == pv.dims || return nothing
    po.dims == pq.dims || return nothing
    pq.dims[1] == pk.dims[1] || return nothing                     # the head width
    pq.dims[3] == pk.dims[3] && pq.dims[4] == pk.dims[4] || return nothing
    po.dims[1] * sizeof(To) % 16 == 0 || return nothing
    return (; keys = map(packedkey, ps), res = map(p -> p.res, ps))
end

"""
    sdpa_batched!(o, qres, kres, vres, qkey, kkey, vkey, scale) -> o

`o = softmax(scale · qᵀk) · v` per `(head, batch)` plane, Apple's fused op, encoded
into the command buffer the current queue is batching into.

The three keys come from [`sdpa_operands`](@ref) and say how each operand lies; the
three arrays are the dense arrays those layouts are windows on, which is what makes
this callable with addresses that were fixed when a plan was recorded. The graph is
built once per (layout, type, scale) and kept.
"""
function sdpa_batched!(o::MtlArray, qres::MtlArray, kres::MtlArray, vres::MtlArray,
                       qkey, kkey, vkey, scale::Real)
    key = SDPAGraphKey(qkey, kkey, vkey, eltype(qres), eltype(o), Float32(scale))
    cached = @lock _sdpa_graph_cache_lock get!(_sdpa_graph_cache, key) do
        CachedSDPAGraph(key)
    end
    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
        cached.qph => packeddata(qres, qkey[2]),
        cached.kph => packeddata(kres, kkey[2]),
        cached.vph => packeddata(vres, vkey[2]),
    )
    results = Dict{MPSGraphTensor,MPSGraphTensorData}(cached.out => tensordata(o))
    encode_batched!(cached.graph, feeds, results, o, qres, kres, vres)
    return o
end
