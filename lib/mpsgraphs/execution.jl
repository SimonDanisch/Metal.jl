
MPS.encode!(commandBuffer::MPSCommandBuffer, graph::MPSGraph, feeds::MPSGraphTensorDataDictionary, resultsDictionary::MPSGraphTensorDataDictionary) = @inline MPS.encode!(commandBuffer, graph, feeds, resultsDictionary, nil, MPSGraphExecutionDescriptor())
function MPS.encode!(commandBuffer::MPSCommandBuffer, graph::MPSGraph, feeds::MPSGraphTensorDataDictionary, resultsDictionary::MPSGraphTensorDataDictionary, targetOperations, executionDescriptor::MPSGraphExecutionDescriptor)
    @objc [graph::id{MPSGraph} encodeToCommandBuffer:commandBuffer::id{MPSCommandBuffer}
                                                  feeds:feeds::id{MPSGraphTensorDataDictionary}
                                       targetOperations:targetOperations::id{Object}
                                      resultsDictionary:resultsDictionary::id{MPSGraphTensorDataDictionary}
                                    executionDescriptor:executionDescriptor::id{MPSGraphExecutionDescriptor}]::Nothing
    return resultsDictionary
end

function MPS.encode!(commandBuffer::MPSCommandBuffer, graph::MPSGraph, feeds::MPSGraphTensorDataDictionary, targetTensors::NSArray, targetOperations=nil, executionDescriptor::MPSGraphExecutionDescriptor=MPSGraphExecutionDescriptor())
    obj = @objc [graph::id{MPSGraph} encodeToCommandBuffer:commandBuffer::id{MPSCommandBuffer}
                                                        feeds:feeds::id{MPSGraphTensorDataDictionary}
                                                targetTensors:targetTensors::id{NSArray}
                                             targetOperations:targetOperations::id{Object}
                                          executionDescriptor:executionDescriptor::id{MPSGraphExecutionDescriptor}]::id{MPSGraphTensorDataDictionary}
    MPSGraphTensorDataDictionary(obj)
end

function run(graph::MPSGraph, feeds::MPSGraphTensorDataDictionary, targetTensors::NSArray, targetOperations=nil)
    obj = @objc [graph::id{MPSGraph} runWithFeeds:feeds::id{MPSGraphTensorDataDictionary}
                                            targetTensors:targetTensors::id{NSArray}
                                         targetOperations:targetOperations::id{Object}]::id{MPSGraphTensorDataDictionary}
    MPSGraphTensorDataDictionary(obj)
end

function run(graph::MPSGraph, commandQueue, feeds::MPSGraphTensorDataDictionary, targetTensors::NSArray)
    Metal.flush!(commandQueue)
    obj = @objc [graph::id{MPSGraph} runWithMTLCommandQueue:commandQueue::id{MTLCommandQueue}
                                                    feeds:feeds::id{MPSGraphTensorDataDictionary}
                                            targetTensors:targetTensors::id{NSArray}
                                         targetOperations:nil::id{Object}]::id{MPSGraphTensorDataDictionary}
    MPSGraphTensorDataDictionary(obj)
end

const MPSGraphTensorShapedTypeDictionary = NSDictionary#{MPSGraphTensor, MPSGraphTensorShapedType}

compile(graph::MPSGraph, dev::MTLDevice, feeds::MPSGraphTensorShapedTypeDictionary, targetTensors::NSArray, targetOperations=nil, compilationDescriptor=nil) = compile(graph, MPSGraphDevice(dev), feeds, targetTensors, targetOperations, compilationDescriptor)
function compile(graph::MPSGraph, dev::MPSGraphDevice, feeds::MPSGraphTensorShapedTypeDictionary, targetTensors::NSArray, targetOperations=nil, compilationDescriptor=nil)
    return @objc [graph::id{MPSGraph} compileWithDevice:dev::id{MPSGraphDevice}
                                      feeds:feeds::id{MPSGraphTensorShapedTypeDictionary}
                              targetTensors:targetTensors::id{NSArray}
                           targetOperations:targetOperations::id{Object}
                      compilationDescriptor:compilationDescriptor::id{Object}]::MPSGraphExecutable
end

function MPSGraphExecutableSerializationDescriptor()
    return @objc [MPSGraphExecutableSerializationDescriptor alloc]::MPSGraphExecutableSerializationDescriptor
end

serialize(graphExe::MPSGraphExecutable, url, descriptor=MPSGraphExecutableSerializationDescriptor()) = serialize(graphExe, NSFileURL(url), descriptor)
function serialize(graphExe::MPSGraphExecutable, url::NSURL, descriptor=MPSGraphExecutableSerializationDescriptor())
    @objc [graphExe::id{MPSGraphExecutable} serializeToMPSGraphPackageAtURL:url::id{NSURL}
                              descriptor:descriptor::id{MPSGraphExecutableSerializationDescriptor}]::Nothing
end

"""
    encode_batched!(graph, feeds, results, roots...)

Encode `graph` into the command buffer the current queue is already BATCHING into,
rather than committing one of its own.

Every other entry point here builds an `MPSCommandBuffer` off the queue and commits
it. That is right in isolation and wrong in company: command buffers execute in
COMMIT order, so an MPS op that commits its own while a batch of kernel launches is
still open runs BEFORE them. Joining the open buffer makes the op ordered by encoder
order instead — the same rule the launches already rely on — and costs no extra
submission.

`roots` are kept alive until that command buffer retires. An `MPSGraphTensorData`
holds the `MTLBuffer` but nothing tells Julia the `MtlArray` is still in use.
"""
function encode_batched!(graph::MPSGraph, feeds, results, roots...)
    bq = Metal.global_queue(Metal.device())
    # The open encoder has to END first: a command buffer may have only one encoder
    # at a time and MPS makes its own. Ending is not committing — the buffer stays
    # open and the next launch gets a fresh encoder in it.
    Metal.end_encoder!(bq)
    cmdbuf = Metal.ensure_cmdbuf!(bq)
    mps = MPSCommandBuffer(cmdbuf)
    encode!(mps, graph, NSDictionary(feeds), NSDictionary(results), nil,
            default_exec_desc())
    # MPS may have `commitAndContinue`d: committed the buffer it was given and moved
    # to one of its own, which it does on its own schedule — once in 161 encodes of
    # SAM 2.1's encoder frame. The batch adopts the continuation rather than being
    # left holding a committed buffer. A no-op in the usual case.
    Metal.adopt_continued!(bq, cmdbuf, mps.commandBuffer)
    Metal.record_operation!(bq, roots...)
    return nothing
end

"""
    tensordata(arr::MtlArray) -> MPSGraphTensorData

An array's bytes as MPS sees them, HONOURING its offset.

`MPSGraphTensorData(::MtlArray)` binds `arr.data[]` — the whole buffer — and the
offset is silently dropped, which is correct only for an array that starts one. A
suballocated array (every transient of a render graph is a slice of a 64 MiB block)
reads and writes the wrong bytes. `MPSNDArray` takes the offset, so the route
through it is the one that works for both.
"""
# The buffer form where it is correct, which is every array that starts one: it takes
# any shape, while `MPSNDArray` pads the innermost row to sixteen bytes. The offset
# route is for the arrays that need it.
#
# `shape` is what MPS is TOLD, which need not be the array's own — see `bindshape`: an
# operand whose innermost extent is not a multiple of sixteen bytes is bound flat and
# reshaped in the graph, which is the only way to hand MPS a `7x7x3x96` half weight at
# a nonzero offset at all.
function tensordata(arr::Metal.MtlArray, shape::Tuple = size(arr))
    arr.offset == 0 && return MPSGraphTensorData(arr.data[],
        convert(MPSShape, reverse(shape)), eltype(arr))
    desc = MPS.MPSNDArrayDescriptor(eltype(arr), collect(shape))
    return MPSGraphTensorData(MPS.MPSNDArray(arr.data[], UInt(arr.offset), desc))
end

"""
    bindshape(dims, T) -> Tuple or nothing

The shape to hand MPS for a DENSE operand whose logical shape is `dims`.

`MPSNDArray` created over a buffer pads the innermost row to sixteen bytes and then
refuses a buffer that is not big enough for the padded layout — a `(7, 7, 3, 8)` half
array is 2352 bytes and it asks for 2688. A convolution weight is never 16-byte wide
in its kernel extent, so such an operand is bound FLAT and reshaped in the graph,
which is free and always aligned when the whole array is.

`nothing` when neither shape works, which is how a caller learns to keep its own
kernel.
"""
function bindshape(dims::Tuple, T::DataType)
    first(dims) * sizeof(T) % 16 == 0 && return dims
    n = prod(dims)
    n * sizeof(T) % 16 == 0 && return (n,)
    return nothing
end

"""Bind a placeholder built on `bindshape` back to the logical shape."""
reshapebound(graph::MPSGraph, t::MPSGraphTensor, bound::Tuple, dims::Tuple, name) =
    bound === dims ? t :
    reshapeTensor(graph, t, convert(MPSShape, reverse(dims)), name)
