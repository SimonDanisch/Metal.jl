export MTLComputeCommandEncoder
export set_function!, set_buffer!, set_bytes!, dispatchThreadgroups!, endEncoding!
export dispatchThreadgroupsIndirect!
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

# A table is bound at a BUFFER index — the shader declares it as a parameter with
# `[[buffer(n)]]` like any other, and Metal resolves the function pointers behind
# it. It is also a `MTLResource`, so it needs `useResource!` to be resident just
# as a buffer reached by address does.
function set_visible_function_table!(cce::MTLComputeCommandEncoder,
                                     table::MTLVisibleFunctionTable, index::Integer)
    @objc [cce::id{MTLComputeCommandEncoder} setVisibleFunctionTable:table::id{MTLVisibleFunctionTable} atBufferIndex:index::NSUInteger]::Nothing
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

"""
    dispatchThreadgroupsIndirect!(cce, buf, offset, threadsPerThreadgroup)

Dispatch with the threadgroup count read from `buf` on the DEVICE.

`buf` holds three `UInt32` at `offset` — the grid in x, y, z — which is the same
`MTLDispatchThreadgroupsIndirectArguments` layout Metal documents, and the same
shape as the indirect draw arguments the render encoder takes.

The point is what does NOT happen: the host never learns the count. A dispatch
whose size a previous kernel computed otherwise has to be sized on the CPU,
which means reading device memory, which means waiting for that kernel — a full
queue drain in the middle of a frame. Here the count stays on the device and the
two dispatches are just ordered on the queue.

The count is in THREADGROUPS, not threads: whatever writes it has to divide by
the threadgroup width itself, because this call cannot.
"""
function dispatchThreadgroupsIndirect!(cce::MTLComputeCommandEncoder, buf::MTLBuffer,
                                       offset::Integer, threadsPerThreadgroup::MTLSize)
    @objc [cce::id{MTLComputeCommandEncoder} dispatchThreadgroupsWithIndirectBuffer:buf::id{MTLBuffer}
                                             indirectBufferOffset:offset::NSUInteger
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

"""
`useResources:` over an ALREADY-MARSHALLED array of object pointers.

The `Vector{MTLBuffer}` method above converts through `Base.cconvert`, which
builds a fresh `Vector{id}` and an `idArray` to hold it on EVERY call. A caller
whose resource list is stable across frames — which is the point of caching such
a list — therefore still pays that conversion once a frame for an answer that
cannot have changed. Converting once and passing the result here costs nothing
per call.

The caller owns the array and owns keeping the objects it points at alive: these
are bare pointers and nothing here roots them.
"""
function use!(cce::MTLComputeCommandEncoder, ids::Vector{id{MTLBuffer}},
              mode::MTLResourceUsage=ReadWriteUsage)
    @objc [cce::id{MTLComputeCommandEncoder} useResources:ids::Ptr{id{MTLBuffer}}
                                             count:length(ids)::Csize_t
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
