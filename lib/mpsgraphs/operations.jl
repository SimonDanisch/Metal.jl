
function broadcastTensor(graph::MPSGraph, tensor::MPSGraphTensor, shape::MPSShape, name="broadcast")
    @objc [graph::id{MPSGraph} broadcastTensor:tensor::id{MPSGraphTensor}
                                      toShape:shape::id{MPSShape}
                                         name:name::id{NSString}]::MPSGraphTensor
end
function broadcastTensor(graph::MPSGraph, tensor::MPSGraphTensor, shapeTensor::MPSGraphTensor, name="broadcast")
    @objc [graph::id{MPSGraph} broadcastTensor:tensor::id{MPSGraphTensor}
                                 toShapeTensor:shapeTensor::id{MPSGraphTensor}
                                           name:name::id{NSString}]::MPSGraphTensor
end

function mps_axis(ndims::Integer, dim::Integer)
    1 <= dim <= ndims || throw(ArgumentError("dimension out of range"))
    return ndims - dim
end

mps_axis(shape::Tuple, dim::Integer) = mps_axis(length(shape), dim)

function check_mpsgraph_offsets(arrays::MtlArray...)
    all(A -> A.offset == 0, arrays) ||
        throw(ArgumentError("MPSGraph operations require zero-offset Metal arrays"))
    return nothing
end

function castTensor(graph::MPSGraph, tensor::MPSGraphTensor, toType, name = "cast")
    @objc [graph::id{MPSGraph} castTensor:tensor::id{MPSGraphTensor}
                                    toType:toType::MPSDataType
                                      name:name::id{NSString}]::MPSGraphTensor
end

function reshapeTensor(graph::MPSGraph, tensor::MPSGraphTensor, shape::MPSShape,
                       name = "reshape")
    @objc [graph::id{MPSGraph} reshapeTensor:tensor::id{MPSGraphTensor}
                                   withShape:shape::id{MPSShape}
                                        name:name::id{NSString}]::MPSGraphTensor
end

# uses the swift name
complexConstant(graph::MPSGraph, n::Number, dataType) = complexConstant(graph::MPSGraph, reim(n)..., dataType)
function complexConstant(graph::MPSGraph, realPart::Number, imaginaryPart::Number, dataType)
    @objc [graph::id{MPSGraph} constantWithRealPart:realPart::Float64
                                      imaginaryPart:imaginaryPart::Float64
                                            dataType:dataType::MPSDataType]::MPSGraphTensor
end

function constantWithScalar(graph::MPSGraph, scalar::Number, dataType)
    @objc [graph::id{MPSGraph} constantWithScalar:scalar::Float64
                                         dataType:dataType::MPSDataType]::MPSGraphTensor
end

function matrixMultiplicationWithPrimaryTensor(graph::MPSGraph, primary::MPSGraphTensor, secondary::MPSGraphTensor, name = "matmul")
    @objc [graph::id{MPSGraph} matrixMultiplicationWithPrimaryTensor:primary::id{MPSGraphTensor}
                                                     secondaryTensor:secondary::id{MPSGraphTensor}
                                                                 name:name::id{NSString}]::MPSGraphTensor
end

function multiplicationWithPrimaryTensor(graph::MPSGraph, primary::MPSGraphTensor, secondary::MPSGraphTensor, name = "mul")
    @objc [graph::id{MPSGraph} multiplicationWithPrimaryTensor:primary::id{MPSGraphTensor}
                                               secondaryTensor:secondary::id{MPSGraphTensor}
                                                           name:name::id{NSString}]::MPSGraphTensor
end
function additionWithPrimaryTensor(graph::MPSGraph, primary::MPSGraphTensor, secondary::MPSGraphTensor, name = "add")
    @objc [graph::id{MPSGraph} additionWithPrimaryTensor:primary::id{MPSGraphTensor}
                                         secondaryTensor:secondary::id{MPSGraphTensor}
                                                     name:name::id{NSString}]::MPSGraphTensor
end

function transposeTensor(graph::MPSGraph, tensor::MPSGraphTensor, dimension, withDimension, name = "transpose")
    @objc [graph::id{MPSGraph} transposeTensor:tensor::id{MPSGraphTensor}
                                     dimension:dimension::NSUInteger
                                 withDimension:withDimension::NSUInteger
                                           name:name::id{NSString}]::MPSGraphTensor
end

"""
    sliceTensor(graph, tensor, dimension, start, length) -> MPSGraphTensor

`length` entries of `dimension` starting at `start`. `dimension` is an MPSGraph axis
(outermost first, zero-based) — see [`mps_axis`](@ref) for the Julia one. The sliced
axis is KEPT, with the length given, so axis numbers after this do not move.
"""
function sliceTensor(graph::MPSGraph, tensor::MPSGraphTensor, dimension, start, len,
                     name = "slice")
    @objc [graph::id{MPSGraph} sliceTensor:tensor::id{MPSGraphTensor}
                                 dimension:dimension::NSUInteger
                                     start:start::NSInteger
                                    length:len::NSInteger
                                      name:name::id{NSString}]::MPSGraphTensor
end

function reLUWithTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "relu")
    @objc [graph::id{MPSGraph} reLUWithTensor:tensor::id{MPSGraphTensor}
                                        name:name::id{NSString}]::MPSGraphTensor
end

function erfWithTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "erf")
    @objc [graph::id{MPSGraph} erfWithTensor:tensor::id{MPSGraphTensor}
                                       name:name::id{NSString}]::MPSGraphTensor
end

function tanhWithTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "tanh")
    @objc [graph::id{MPSGraph} tanhWithTensor:tensor::id{MPSGraphTensor}
                                        name:name::id{NSString}]::MPSGraphTensor
end

function shapeOfTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "shapeOfTensor")
    @objc [graph::id{MPSGraph} shapeOfTensor:tensor::id{MPSGraphTensor}
                                        name:name::id{NSString}]::MPSGraphTensor
end

function identityWithTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "identity")
    @objc [graph::id{MPSGraph} identityWithTensor:tensor::id{MPSGraphTensor}
                                             name:name::id{NSString}]::MPSGraphTensor
end

function conjugateWithTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "conjugate")
    @objc [graph::id{MPSGraph} conjugateWithTensor:tensor::id{MPSGraphTensor}
                                              name:name::id{NSString}]::MPSGraphTensor
end

function negativeWithTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "negate")
    @objc [graph::id{MPSGraph} negativeWithTensor:tensor::id{MPSGraphTensor}
                                             name:name::id{NSString}]::MPSGraphTensor
end

function imaginaryPartOfTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "imaginarypart")
    @objc [graph::id{MPSGraph} imaginaryPartOfTensor:tensor::id{MPSGraphTensor}
                                                name:name::id{NSString}]::MPSGraphTensor
end

function realPartOfTensor(graph::MPSGraph, tensor::MPSGraphTensor, name = "realpart")
    @objc [graph::id{MPSGraph} realPartOfTensor:tensor::id{MPSGraphTensor}
                                           name:name::id{NSString}]::MPSGraphTensor
end

function complexTensorWithRealTensor(graph::MPSGraph, realTensor::MPSGraphTensor, imaginaryTensor::MPSGraphTensor, name="complex")
    @objc [graph::id{MPSGraph} complexTensorWithRealTensor:realTensor::id{MPSGraphTensor}
                                           imaginaryTensor:imaginaryTensor::id{MPSGraphTensor}
                                                       name:name::id{NSString}]::MPSGraphTensor
end

function scaledDotProductAttentionWithQueryTensor(graph::MPSGraph, Q::MPSGraphTensor,
                                                  K::MPSGraphTensor, V::MPSGraphTensor,
                                                  scale::Real, name = "sdpa")
    @objc [graph::id{MPSGraph} scaledDotProductAttentionWithQueryTensor:Q::id{MPSGraphTensor}
                                                              keyTensor:K::id{MPSGraphTensor}
                                                            valueTensor:V::id{MPSGraphTensor}
                                                                  scale:scale::Cfloat
                                                                   name:name::id{NSString}]::MPSGraphTensor
end
function scaledDotProductAttentionWithQueryTensor(graph::MPSGraph, Q::MPSGraphTensor,
                                                  K::MPSGraphTensor, V::MPSGraphTensor,
                                                  mask::MPSGraphTensor, scale::Real,
                                                  name = "sdpa")
    @objc [graph::id{MPSGraph} scaledDotProductAttentionWithQueryTensor:Q::id{MPSGraphTensor}
                                                              keyTensor:K::id{MPSGraphTensor}
                                                            valueTensor:V::id{MPSGraphTensor}
                                                             maskTensor:mask::id{MPSGraphTensor}
                                                                  scale:scale::Cfloat
                                                                   name:name::id{NSString}]::MPSGraphTensor
end
