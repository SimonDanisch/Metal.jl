#
# compute pipeline descriptor
#

export MTLComputePipelineDescriptor

# @objcwrapper managed = true MTLComputePipelineDescriptor <: NSObject

function MTLComputePipelineDescriptor()
    return @objc [MTLComputePipelineDescriptor new]::MTLComputePipelineDescriptor
end

export MTLLinkedFunctions

"""
    MTLLinkedFunctions()

An empty set of functions to link into a pipeline.

Assign `.functions` (a `Vector{MTLFunction}` of `[[visible]]` functions) and set
it on a `MTLComputePipelineDescriptor.linkedFunctions`. This is how a kernel
calls something the Metal frontend had to compile — an `intersector<>`, for
instance — that has no AIR symbol a compiler back-end could emit directly.
"""
function MTLLinkedFunctions()
    return @objc [MTLLinkedFunctions new]::MTLLinkedFunctions
end

#
# compute pipeline state
#

export MTLComputePipelineState

# @objcwrapper managed = true MTLComputePipelineState <: NSObject

function MTLComputePipelineState(dev::MTLDevice, fun::MTLFunction)
    err = Ref{id{NSError}}(nil)
    pipeline = @objc [dev::id{MTLDevice} newComputePipelineStateWithFunction:fun::id{MTLFunction}
                                          error:err::Ptr{id{NSError}}]::Union{Nothing,MTLComputePipelineState}
    pipeline === nothing && throw_error(err[])

    return pipeline
end

# Descriptor-based creation. Needed to attach binary archives (`desc.binaryArchives`) and
# to pass `MTLPipelineOption`s such as `MTLPipelineOptionFailOnBinaryArchiveMiss`. Reflection
# is skipped (unreliable on archive hits); pass a null out-param.
function MTLComputePipelineState(dev::MTLDevice, desc::MTLComputePipelineDescriptor;
                                 options::MTLPipelineOption=MTLPipelineOptionNone)
    err = Ref{id{NSError}}(nil)
    pipeline = @objc [dev::id{MTLDevice} newComputePipelineStateWithDescriptor:desc::id{MTLComputePipelineDescriptor}
                                          options:options::MTLPipelineOption
                                          reflection:C_NULL::Ptr{Cvoid}
                                          error:err::Ptr{id{NSError}}]::Union{Nothing,MTLComputePipelineState}
    pipeline === nothing && throw_error(err[])

    return pipeline
end
