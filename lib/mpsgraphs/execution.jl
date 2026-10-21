
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
    return @objc [graph::id{MPSGraph} encodeToCommandBuffer:commandBuffer::id{MPSCommandBuffer}
                                                         feeds:feeds::id{MPSGraphTensorDataDictionary}
                                                 targetTensors:targetTensors::id{NSArray}
                                              targetOperations:targetOperations::id{Object}
                                           executionDescriptor:executionDescriptor::id{MPSGraphExecutionDescriptor}]::MPSGraphTensorDataDictionary
end

function run(graph::MPSGraph, feeds::MPSGraphTensorDataDictionary, targetTensors::NSArray, targetOperations=nil)
    return @objc [graph::id{MPSGraph} runWithFeeds:feeds::id{MPSGraphTensorDataDictionary}
                                             targetTensors:targetTensors::id{NSArray}
                                          targetOperations:targetOperations::id{Object}]::MPSGraphTensorDataDictionary
end

function run(graph::MPSGraph, commandQueue, feeds::MPSGraphTensorDataDictionary, targetTensors::NSArray)
    Metal.flush!(commandQueue)
    # this executes synchronously, without going through Metal.jl's submission hook
    Metal.synchronize_pending!()
    return @objc [graph::id{MPSGraph} runWithMTLCommandQueue:commandQueue::id{MTLCommandQueue}
                                                     feeds:feeds::id{MPSGraphTensorDataDictionary}
                                             targetTensors:targetTensors::id{NSArray}
                                          targetOperations:nil::id{Object}]::MPSGraphTensorDataDictionary
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


## cached graphs

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

function MPS.encode!(commandBuffer::MPSCommandBuffer, executable::MPSGraphExecutable,
                     inputs::NSArray, results::NSArray, executionDescriptor=nil)
    @objc [executable::id{MPSGraphExecutable} encodeToCommandBuffer:commandBuffer::id{MPSCommandBuffer}
                                                         inputsArray:inputs::id{NSArray}
                                                        resultsArray:results::id{NSArray}
                                                 executionDescriptor:executionDescriptor::id{Object}]::id{NSArray}
    return results
end

# Graphs are built and compiled once per configuration and device, and reused; doing so
# dominates the run time of small operations. Executing a compiled graph also avoids the
# per-call lookup that MPSGraph performs to find its own compiled version.

"""
    CachedGraph(graph, inputs, outputs; constants=[])

Compile `graph` for the current device, given the placeholders it reads (`inputs`) and
the tensors it produces (`outputs`), in the order in which [`run!`](@ref) binds arrays to
them. `constants` pairs further placeholders with arrays that are bound on every run;
unlike graph constants, which MPSGraph converts through Float64, they hold any value
exactly.
"""
struct CachedGraph
    executable::MPSGraphExecutable
    # for each input of the executable, which it orders itself, the position in the
    # inputs passed to `run!` followed by the constants
    input_order::Vector{Int}
    noutputs::Int
    constants::Vector{MtlArray}
    # concurrent encoding of one executable is not documented to be safe
    lock::ReentrantLock
end

function CachedGraph(graph::MPSGraph, inputs::Vector{MPSGraphTensor},
                     outputs::Vector{MPSGraphTensor};
                     constants::Vector{<:Pair{MPSGraphTensor}}=Pair{MPSGraphTensor,MtlArray}[])
    placeholders = [inputs; first.(constants)]
    feeds = Dict(x => MPSGraphShapedType(x.shape, x.dataType) for x in placeholders)
    executable = compile(graph, device(), NSDictionary(feeds), NSArray(outputs), nil,
                         default_exec_desc().compilationDescriptor)
    input_order = map(executable.feedTensors) do x
        something(findfirst(y -> pointer(y) == pointer(x), placeholders))
    end
    return CachedGraph(executable, input_order, length(outputs), last.(constants),
                       ReentrantLock())
end

"""
    build_graph(key)::CachedGraph

Build the graph that `key` describes. Each operation defines a key type, holding everything
that determines the graph's structure (shapes, element types, flags), and a method for it.
"""
function build_graph end

const graph_cache = Dict{Any, CachedGraph}()
const graph_cache_lock = ReentrantLock()

function cached_graph(key)
    @lock graph_cache_lock get!(() -> build_graph(key), graph_cache, (device(), key))
end

"""
    run!(graph::CachedGraph, outputs, inputs)

Encode `graph` into the command buffer the current task's queue is batching into (see
[`encode_batched!`](@ref)), reading the arrays in `inputs` and writing those in `outputs`.
"""
function run!(cached::CachedGraph, outputs::Tuple, inputs::Tuple)
    ninputs = length(cached.input_order) - length(cached.constants)
    length(inputs) == ninputs && length(outputs) == cached.noutputs ||
        throw(ArgumentError("graph takes $ninputs inputs and $(cached.noutputs) outputs"))
    # MPS aborts the process on empty buffers; operations must handle empty arrays
    (any(isempty, inputs) || any(isempty, outputs)) &&
        throw(ArgumentError("cannot run a graph on empty arrays"))
    all_inputs = (inputs..., cached.constants...)
    input_data = NSArray([MPSGraphTensorData(all_inputs[i]) for i in cached.input_order])
    output_data = NSArray([MPSGraphTensorData(arr) for arr in outputs])

    # Into the OPEN batch rather than a command buffer of its own, as `encode_batched!`
    # does and for the same reason: no submission of its own, and ordered against the
    # launches around it by encoder order.
    bq = Metal.global_queue(device())
    Metal.joingraph!(bq, cached.executable)
    Metal.batchbuffer(all_inputs..., outputs...) do cmdbuf
        mps = MPSCommandBuffer(cmdbuf)
        @lock cached.lock begin
            encode!(mps, cached.executable, input_data, output_data)
        end
        Metal.adopt_continued!(bq, cmdbuf, mps.commandBuffer)
        Metal.joingraph!(bq, cached.executable)
    end
    return
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
    # `batchbuffer` ends the open encoder and holds the operands; see its docstring for
    # why a command buffer of this call's own would be wrong.
    bq = Metal.global_queue(Metal.device())
    # Before the buffer is taken: it may commit the batch to make room (a command
    # buffer holds the residency sets of at most 16 distinct graphs).
    Metal.joingraph!(bq, graph)
    # The encode, the adoption and the count all inside `batchbuffer`'s lock: a launch
    # or a flush from another task between them lands in a buffer MPS is encoding into.
    Metal.batchbuffer(roots...) do cmdbuf
        mps = MPSCommandBuffer(cmdbuf)
        encode!(mps, graph, NSDictionary(feeds), NSDictionary(results), nil,
                default_exec_desc())
        # MPS may have `commitAndContinue`d: committed the buffer it was given and
        # moved to one of its own, which it does on its own schedule — once in 161
        # encodes of SAM 2.1's encoder frame. The batch adopts the continuation rather
        # than being left holding a committed buffer. A no-op in the usual case.
        Metal.adopt_continued!(bq, cmdbuf, mps.commandBuffer)
        # An adopted continuation starts with no graphs counted, and holds this one.
        Metal.joingraph!(bq, graph)
    end
    return nothing
end

"""
    tensordata(arr::MtlArray) -> MPSGraphTensorData

An array's bytes as MPS sees them, HONOURING its offset.

`MPSGraphTensorData(::MtlArray)` binds the array's whole buffer, and the
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
    arr.offset == 0 && return MPSGraphTensorData(Base.unsafe_convert(MTLBuffer, arr),
        convert(MPSShape, reverse(shape)), eltype(arr))
    desc = MPS.MPSNDArrayDescriptor(eltype(arr), collect(shape))
    return MPSGraphTensorData(MPS.MPSNDArray(Base.unsafe_convert(MTLBuffer, arr), UInt(arr.offset), desc))
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
