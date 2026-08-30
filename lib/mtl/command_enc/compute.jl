export MTLComputeCommandEncoder
export set_function!, set_buffer!, set_bytes!, dispatchThreadgroups!, endEncoding!
export set_acceleration_structure!
export append_current_function!

# @objcwrapper MTLComputeCommandEncoder <: MTLCommandEncoder

function MTLComputeCommandEncoder(cmdbuf::MTLCommandBuffer;
                                  dispatch_type::Union{Nothing,MTLDispatchType} = nothing)
    if isnothing(dispatch_type)
        @objc [cmdbuf::id{MTLCommandBuffer} computeCommandEncoder]::MTLComputeCommandEncoder
    else
        @objc [cmdbuf::id{MTLCommandBuffer} computeCommandEncoderWithDispatchType:dispatch_type::MTLDispatchType]::MTLComputeCommandEncoder
    end
end

function set_function!(cce::MTLComputeCommandEncoder, pip::MTLComputePipelineState)
    @objc [cce::id{MTLComputeCommandEncoder} setComputePipelineState:pip::id{MTLComputePipelineState}]::Nothing
end

function set_buffer!(cce::MTLComputeCommandEncoder, buf::MTLBuffer, offset, index)
    @objc [cce::id{MTLComputeCommandEncoder} setBuffer:buf::id{MTLBuffer}
                                             offset:offset::NSUInteger
                                             atIndex:(index-1)::NSUInteger]::Nothing
end

function set_bytes!(cce::MTLComputeCommandEncoder, ptr::Ptr, len::Integer, index::Integer)
    @objc [cce::id{MTLComputeCommandEncoder} setBytes:ptr::Ptr{Cvoid}
                                           length:len::NSUInteger
                                          atIndex:(index-1)::NSUInteger]::Nothing
end

function dispatchThreadgroups!(cce::MTLComputeCommandEncoder, threadgroupsPerGrid, threadsPerThreadgroup)
    @objc [cce::id{MTLComputeCommandEncoder} dispatchThreadgroups:threadgroupsPerGrid::MTLSize
                                             threadsPerThreadgroup:threadsPerThreadgroup::MTLSize]::Nothing
end

function dispatchThreads!(cce::MTLComputeCommandEncoder, threadsPerGrid::MTLSize, threadsPerThreadgroup::MTLSize)
    @objc [cce::id{MTLComputeCommandEncoder} dispatchThreads:threadsPerGrid::MTLSize
                                             threadsPerThreadgroup:threadsPerThreadgroup::MTLSize]::Nothing
end

#####
# encode in the Command Encoder

function MTLComputeCommandEncoder(f::Base.Callable, cmdbuf::MTLCommandBuffer; kwargs...)
    encoder = MTLComputeCommandEncoder(cmdbuf; kwargs...)
    try
        f(encoder)
    finally
        close(encoder)
    end
end

function append_current_function!(cce::MTLComputeCommandEncoder, threadgroupsPerGrid, threadsPerThreadgroup)
    dispatchThreadgroups!(cce, threadgroupsPerGrid, threadsPerThreadgroup)
end

#### use

function use!(cce::MTLComputeCommandEncoder, buf::MTLBuffer, mode::MTLResourceUsage=ReadWriteUsage)
    @objc [cce::id{MTLComputeCommandEncoder} useResource:buf::id{MTLBuffer}
                                             usage:mode::MTLResourceUsage]::Nothing
end

function use!(cce::MTLComputeCommandEncoder, buf::Vector{MTLBuffer}, mode::MTLResourceUsage=ReadWriteUsage)
    @objc [cce::id{MTLComputeCommandEncoder} useResources:buf::id{MTLBuffer}
                                             count:length(buf)::Csize_t
                                             usage:mode::MTLResourceUsage]::Nothing
end

#### acceleration structures

"""
    set_acceleration_structure!(cce, accel, index)

Bind `accel` to `[[buffer(index)]]` for the next dispatch.

An acceleration structure occupies a buffer binding slot even though it is not
a buffer, which is why this is `setAccelerationStructure:atBufferIndex:` and not
a separate table: in MSL the kernel parameter is declared
`instance_acceleration_structure [[buffer(n)]]`, and `n` is this `index`.

`index` is 1-based here and 0-based in Metal, matching `set_buffer!` next door.
"""
function set_acceleration_structure!(cce::MTLComputeCommandEncoder,
                                     accel::MTLAccelerationStructure, index::Integer)
    @objc [cce::id{MTLComputeCommandEncoder} setAccelerationStructure:accel::id{MTLAccelerationStructure}
                                             atBufferIndex:(index - 1)::NSUInteger]::Nothing
end

"""
    use!(cce, accel, mode)

Make `accel` resident for the dispatch.

Required for the BLASes an instance structure points at: the TLAS holds
references the encoder cannot see, so binding the TLAS alone leaves its
instanced structures non-resident and traversal reads garbage. Metal's
validation layer reports this; without it the symptom is missed hits.
"""
function use!(cce::MTLComputeCommandEncoder, accel::MTLAccelerationStructure,
              mode::MTLResourceUsage=ReadUsage)
    @objc [cce::id{MTLComputeCommandEncoder} useResource:accel::id{MTLAccelerationStructure}
                                             usage:mode::MTLResourceUsage]::Nothing
end

function use!(cce::MTLComputeCommandEncoder, accels::Vector{MTLAccelerationStructure},
              mode::MTLResourceUsage=ReadUsage)
    isempty(accels) && return
    @objc [cce::id{MTLComputeCommandEncoder} useResources:accels::id{MTLAccelerationStructure}
                                             count:length(accels)::Csize_t
                                             usage:mode::MTLResourceUsage]::Nothing
end
