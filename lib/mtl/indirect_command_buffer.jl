export MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor
export indirect_compute_command, set_pipeline!, set_kernel_buffer!,
       dispatch_threadgroups!, set_barrier!, reset_command!, reset_range!,
       execute_commands!, execute_commands_indirect!

# An MTLCommandBuffer is SINGLE-USE. That is the whole reason this file exists.
#
# Vulkan's baked graph records a `VkCommandBuffer` once and re-submits it every
# frame; Metal has no such thing — a committed command buffer cannot be committed
# again, so "record once, replay" cannot be built out of one. The replayable unit
# here is the INDIRECT command buffer: commands are encoded into it once, by the
# host or by a kernel, and a normal encoder replays the whole range with
# `executeCommandsInBuffer:withRange:` for as many frames as the plan lives.
#
# `libmtl.jl` (generated) already has the TYPES — the descriptor, the buffer, the
# compute command, the execution range. What was missing is every method that
# makes them do anything, which is what follows.

"""
    MTLIndirectCommandBufferDescriptor(; command_types, max_kernel_buffers,
                                       inherit_pipeline, inherit_buffers,
                                       ray_tracing)

What an indirect command buffer is allowed to hold.

A command may only set what the descriptor declared: `max_kernel_buffers` is the
highest buffer index any encoded command binds, and encoding past it is a driver
error rather than a Julia one. `inherit_pipeline`/`inherit_buffers` say whether a
command takes the encoder's state instead of carrying its own — false for a plan,
which is the point: the recording is complete and the encoder that executes it
holds nothing.
"""
function MTLIndirectCommandBufferDescriptor(;
        command_types::MTLIndirectCommandType = MTLIndirectCommandTypeConcurrentDispatch,
        max_kernel_buffers::Integer = 8,
        inherit_pipeline::Bool = false,
        inherit_buffers::Bool = false,
        ray_tracing::Bool = false)
    desc = @objc [MTLIndirectCommandBufferDescriptor alloc]::id{MTLIndirectCommandBufferDescriptor}
    desc = @objc [desc::id{MTLIndirectCommandBufferDescriptor} init]::MTLIndirectCommandBufferDescriptor
    desc.commandTypes = command_types
    desc.maxKernelBufferBindCount = max_kernel_buffers
    desc.inheritPipelineState = inherit_pipeline
    desc.inheritBuffers = inherit_buffers
    desc.supportRayTracing = ray_tracing
    return desc
end

"""
    MTLIndirectCommandBuffer(device, descriptor, max_commands; storage)

Storage for `max_commands` encoded commands.

`PrivateStorage` because nothing on the host reads these back; the commands are
written through the command objects below, which the driver encodes for the GPU
in its own layout rather than as bytes anyone here can lay out.
"""
function MTLIndirectCommandBuffer(dev::MTLDevice, desc::MTLIndirectCommandBufferDescriptor,
                                  max_commands::Integer;
                                  storage::MTLResourceOptions = MTLResourceStorageModePrivate)
    icb = @objc [dev::id{MTLDevice} newIndirectCommandBufferWithDescriptor:desc::id{MTLIndirectCommandBufferDescriptor}
                                    maxCommandCount:max_commands::NSUInteger
                                    options:storage::MTLResourceOptions]::Union{Nothing,MTLIndirectCommandBuffer}
    icb === nothing && error("could not create an indirect command buffer for " *
                             "$max_commands commands; check the descriptor's limits")
    return icb
end

"""
    indirect_compute_command(icb, index) -> MTLIndirectComputeCommand

The command at `index`, ONE-BASED, to encode into.

One-based like every other index in this wrapper — `set_buffer!` and friends all
subtract one on the way to Objective-C, and a command slot is no different.
"""
function indirect_compute_command(icb::MTLIndirectCommandBuffer, index::Integer)
    @objc [icb::id{MTLIndirectCommandBuffer} indirectComputeCommandAtIndex:(index-1)::NSUInteger]::MTLIndirectComputeCommand
end

function set_pipeline!(cmd::MTLIndirectComputeCommand, pip::MTLComputePipelineState)
    @objc [cmd::id{MTLIndirectComputeCommand} setComputePipelineState:pip::id{MTLComputePipelineState}]::Nothing
end

function set_kernel_buffer!(cmd::MTLIndirectComputeCommand, buf::MTLBuffer,
                            offset::Integer, index::Integer)
    @objc [cmd::id{MTLIndirectComputeCommand} setKernelBuffer:buf::id{MTLBuffer}
                                              offset:offset::NSUInteger
                                              atIndex:(index-1)::NSUInteger]::Nothing
end

"""
    dispatch_threadgroups!(cmd, threadgroups, threads_per_threadgroup)

Give the command its grid. Encoded into the command, not taken from the encoder:
a recorded plan carries its own sizes.
"""
function dispatch_threadgroups!(cmd::MTLIndirectComputeCommand, threadgroups::MTLSize,
                                threads::MTLSize)
    @objc [cmd::id{MTLIndirectComputeCommand} concurrentDispatchThreadgroups:threadgroups::MTLSize
                                              threadsPerThreadgroup:threads::MTLSize]::Nothing
end

"""
    set_barrier!(cmd)

Make this command wait for every command before it in the buffer.

`MTLIndirectCommandTypeConcurrentDispatch` says the commands MAY run
concurrently, which is what makes a replayed plan fast and also what would let a
consumer start before its producer. A pass boundary is a barrier.
"""
set_barrier!(cmd::MTLIndirectComputeCommand) =
    @objc [cmd::id{MTLIndirectComputeCommand} setBarrier]::Nothing

"""Clear one command, so a slot can be encoded again."""
reset_command!(cmd::MTLIndirectComputeCommand) =
    @objc [cmd::id{MTLIndirectComputeCommand} reset]::Nothing

"""Clear a ONE-BASED range of commands."""
function reset_range!(icb::MTLIndirectCommandBuffer, range::UnitRange{<:Integer})
    r = NSRange(first(range) - 1, length(range))
    @objc [icb::id{MTLIndirectCommandBuffer} resetWithRange:r::NSRange]::Nothing
end

"""
    execute_commands!(cce, icb, range)

Replay a ONE-BASED range of the buffer's commands.

This is the whole point of the file: the frame's host work is this one call, and
everything it runs was encoded when the plan was recorded.
"""
function execute_commands!(cce::MTLComputeCommandEncoder, icb::MTLIndirectCommandBuffer,
                           range::UnitRange{<:Integer})
    r = NSRange(first(range) - 1, length(range))
    @objc [cce::id{MTLComputeCommandEncoder} executeCommandsInBuffer:icb::id{MTLIndirectCommandBuffer}
                                             withRange:r::NSRange]::Nothing
end

"""
    execute_commands_indirect!(cce, icb, rangebuf, offset)

Replay a range of the buffer's commands that the DEVICE decided.

`rangebuf` holds an `MTLIndirectCommandBufferExecutionRange` at `offset` — two
`UInt32`, a ZERO-BASED location and a length — and the command processor reads it
when the encoder reaches this point rather than when the host encodes it. A
length of zero runs nothing.

This is how a recorded plan keeps a `repeat!` gate: the iteration's commands are
in the buffer once, and a one-thread kernel writes the length the gate implies
just before the range is read. Nothing about the recording changes and the host
never learns whether the iteration ran.
"""
function execute_commands_indirect!(cce::MTLComputeCommandEncoder,
                                    icb::MTLIndirectCommandBuffer,
                                    rangebuf::MTLBuffer, offset::Integer)
    @objc [cce::id{MTLComputeCommandEncoder} executeCommandsInBuffer:icb::id{MTLIndirectCommandBuffer}
                                             indirectBuffer:rangebuf::id{MTLBuffer}
                                             indirectBufferOffset:offset::NSUInteger]::Nothing
end
