# Apple's 2-D convolution, as one graph node.
#
# `convolution2DWithSourceTensor:weightsTensor:descriptor:` is a direct convolution:
# no im2col matrix to materialise, no padded reduction axis, no epilogue pass to sum
# split-K planes. Reaching for it is worth most of what a `7x7x3` stem costs — SAM
# 2.1's is 20.0 MiB of im2col written and read back before the product starts.
#
# The layouts line up without a transposition. A reversed-layout `(W, H, C, N)` is
# MPSGraph's `(N, C, H, W)`, which is its `NCHW`, and `(KW, KH, Cin, Cout)` is
# `(Cout, Cin, KH, KW)`, which is its `OIHW`. Both are what the descriptor is told.

"""What makes two convolution graphs the same graph."""
struct Conv2DGraphKey
    dims_x::NTuple{4,Int}
    dims_w::NTuple{4,Int}
    dims_o::NTuple{4,Int}
    Txw::DataType
    To::DataType
    Tbias::Union{DataType,Nothing}
    stride::NTuple{2,Int}
    pad::NTuple{2,Int}
    dilation::NTuple{2,Int}
    groups::Int
    act::Symbol
end

"""One built convolution graph and the tensors a call binds."""
struct CachedConv2DGraph
    graph::MPSGraph
    place_x::MPSGraphTensor
    place_w::MPSGraphTensor
    place_bias::Union{MPSGraphTensor,Nothing}
    result::MPSGraphTensor
end

function CachedConv2DGraph(key::Conv2DGraphKey)
    graph = MPSGraph()
    bx = bindshape(key.dims_x, key.Txw)
    bw = bindshape(key.dims_w, key.Txw)
    bo = bindshape(key.dims_o, key.To)
    px = placeholderTensor(graph, bx, key.Txw, "x")
    pw = placeholderTensor(graph, bw, key.Txw, "w")
    # `paddingLeft`/`Right` are along X and `Top`/`Bottom` along Y, and ATen's
    # padding is symmetric in each.
    desc = MPSGraphConvolution2DOpDescriptor(
        key.stride, (key.pad[1], key.pad[1], key.pad[2], key.pad[2]),
        key.dilation, key.groups)
    t = convolution2DWithSourceTensor(
        graph, reshapebound(graph, px, bx, key.dims_x, "xnd"),
        reshapebound(graph, pw, bw, key.dims_w, "wnd"), desc)
    # The EPILOGUE in Float32 whenever there is one at all, which is stricter than
    # `gemm_batched!`. A convolution's other lowering here accumulates into a Float32
    # scratch and converts ONCE, in its epilogue pass — so adding the bias in half
    # after MPS has already rounded the product is a second rounding the caller did
    # not have, and SAM 2.1's stem has a bias and no activation.
    wide = (key.act !== :identity || key.Tbias !== nothing) && key.Txw !== Float32
    Te = wide ? Float32 : key.Txw
    wide && (t = castTensor(graph, t, Float32, "wide"))
    place_bias = nothing
    if key.Tbias !== nothing
        Cout = key.dims_o[3]
        place_bias = placeholderTensor(graph, (Cout,), key.Tbias, "bias")
        # One value per output CHANNEL. A `(Cout,)` tensor against MPS's
        # `(N, Cout, H, W)` would broadcast along W, so it is reshaped to the axis
        # it belongs on — `(1, 1, Cout, 1)` reversed, which is `(1, Cout, 1, 1)`.
        b = reshapeTensor(graph, place_bias, convert(MPSShape, [1, Cout, 1, 1]),
                          "biasnchw")
        key.Tbias === Te || (b = castTensor(graph, b, Te, "castbias"))
        t = additionWithPrimaryTensor(graph, t, b, "biasadd")
    end
    t = activation(graph, t, key.act, Te)
    key.To === Te || (t = castTensor(graph, t, key.To, "castout"))
    # …and the destination is bound the same way it is read, so the result tensor has
    # to be the shape MPS was told rather than the logical one.
    t = reshapebound(graph, t, key.dims_o, bo, "outbound")
    return CachedConv2DGraph(graph, px, pw, place_bias, t)
end

const _conv2d_graph_cache = Dict{Conv2DGraphKey,CachedConv2DGraph}()
const _conv2d_graph_cache_lock = ReentrantLock()

"""
    conv2d_shape_supported(out, x, w, bias, act) -> Bool

Whether Apple's convolution covers these operands as they lie.

`x` is `(W, H, Cin, N)`, `w` is `(KW, KH, Cin ÷ groups, Cout)` and `out` is
`(OW, OH, Cout, N)`. One element type for the input and the weight, an activation the
graph has a node for, and a destination no wider than them — `convolution2DWithSource`
has no compute type any more than the product does, so a half convolution asked for in
single is a half convolution cast afterwards.
"""
function conv2d_shape_supported(out, x, w, bias, act::Symbol = :identity)
    act in GEMM_ACTIVATIONS || return false
    all(v -> applicable(ndims, v) && applicable(eltype, v) && applicable(size, v),
        (out, x, w)) || return false
    ndims(out) == ndims(x) == ndims(w) == 4 || return false
    T = eltype(x)
    eltype(w) === T || return false
    (T === Float16 || T === Float32) || return false
    eltype(out) === T || return false
    size(out, 3) == size(w, 4) || return false
    size(out, 4) == size(x, 4) || return false
    for v in (out, x, w)
        bindshape(size(v), eltype(v)) === nothing && return false
    end
    if bias !== nothing
        applicable(ndims, bias) && applicable(eltype, bias) || return false
        ndims(bias) == 1 && length(bias) == size(out, 3) || return false
        # One axis, so there is no flat shape to fall back to: an output channel count
        # whose bytes are not a multiple of sixteen cannot be bound at an offset.
        bindshape(size(bias), eltype(bias)) === nothing && return false
    end
    return true
end

"""
    conv2d_batched!(out, x, w, bias, stride, pad, dilation, groups, act) -> out

`out = act.(conv(x, w) .+ bias)`, Apple's direct convolution, encoded into the command
buffer the current queue is batching into.

Operands are in the reversed layout described by [`conv2d_shape_supported`](@ref) and
may be suballocated. The graph is built once per (shape, type, geometry, activation)
and kept.
"""
function conv2d_batched!(out::MtlArray, x::MtlArray, w::MtlArray, bias,
                         stride::NTuple{2,Int}, pad::NTuple{2,Int},
                         dilation::NTuple{2,Int}, groups::Int,
                         act::Symbol = :identity)
    conv2d_shape_supported(out, x, w, bias, act) || throw(ArgumentError(
        "conv2d_batched!: these operands are not ones MPSGraph can be given — " *
        "$(size(x)) * $(size(w)) into $(size(out)), $(eltype(x))/$(eltype(out)), " *
        "activation :$act. Ask `conv2d_shape_supported` first."))
    key = Conv2DGraphKey(size(x), size(w), size(out), eltype(x), eltype(out),
                         bias === nothing ? nothing : eltype(bias),
                         stride, pad, dilation, groups, act)
    cached = @lock _conv2d_graph_cache_lock get!(_conv2d_graph_cache, key) do
        CachedConv2DGraph(key)
    end
    feeds = Dict{MPSGraphTensor,MPSGraphTensorData}(
        cached.place_x => tensordata(x, bindshape(size(x), eltype(x))),
        cached.place_w => tensordata(w, bindshape(size(w), eltype(w))),
    )
    bias === nothing || (feeds[cached.place_bias] = tensordata(bias))
    results = Dict{MPSGraphTensor,MPSGraphTensorData}(
        cached.result => tensordata(out, bindshape(size(out), eltype(out))))
    if bias === nothing
        encode_batched!(cached.graph, feeds, results, out, x, w)
    else
        encode_batched!(cached.graph, feeds, results, out, x, w, bias)
    end
    return out
end
