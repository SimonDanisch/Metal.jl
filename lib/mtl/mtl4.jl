export MTL4CommandQueue, MTL4CommandAllocator, MTL4CommandBuffer, MTL4ArgumentTable
export MTL4Feedback, supports_mtl4, begin_command_buffer!, end_command_buffer!, compute_encoder,
       use_residency_set!, add_residency_set!, signal_event!, wait_for_event!,
       set_argument_table!, set_address!, barrier!

# Metal 4 submission, which is the model Metal has been moving towards and the one
# a render graph already wanted.
#
# `libmtl.jl` (generated) has the TYPES — the queue, the allocator, the command
# buffer, the compute encoder, the argument table. Every method that makes them do
# anything is here, the same division as `indirect_command_buffer.jl`.
#
# ── What is different, and why it matters here ───────────────────────────────
#
# **A command buffer is reusable.** `MTLCommandBuffer` is single-use: committing
# it consumes it, which is why a replayable recording had to be an
# `MTLIndirectCommandBuffer` in the first place. An `MTL4CommandBuffer` is a
# handle that `beginCommandBufferWithAllocator:` refills, and the allocator is the
# storage it refills from — `MTL4CommandAllocator` is a command pool, and `reset`
# is `vkResetCommandPool`.
#
# **Completion is a timeline, not a callback.** There is no `waitUntilCompleted`
# and no `addCompletedHandler:` on an MTL4 command buffer. The queue signals an
# `MTLSharedEvent` with a value (`signalEvent:value:`) and waits on one
# (`waitForEvent:value:`), which is a timeline semaphore under another name.
# That is what lets a caller ask "has the GPU finished with these bytes" without
# draining the device.
#
# **Three things the compute encoder does not have.** No `setBytes:`, no
# `setBuffer:` — arguments are GPU ADDRESSES written into an `MTL4ArgumentTable`.
# No `useResource:` — residency is residency SETS only. And no automatic hazard
# tracking, so every dependency the driver infers today is an explicit
# `barrier...`. The first is the model a recorded plan already uses (a recorded
# command binds buffer slices by address), which is why replaying one needs none
# of the migration work an ordinary launch does.
#
# Everything here is macOS 26 and later. `supports_mtl4` is the per-device
# question and it is asked once, at device creation.

"""
    supports_mtl4(dev) -> Bool

Whether this device can create an `MTL4CommandQueue`.

Per DEVICE and not per OS: the selector exists on macOS 26, and whether a given
GPU answers `true` is the GPU's business. Asked through `respondsToSelector:`
first so that the question is also answerable on an older system, where sending
it would be an unrecognised selector rather than a `false`.
"""
function supports_mtl4(dev::MTLDevice)
    sel = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "supportsMTL4CommandQueue")
    ccall(:objc_msgSend, Bool, (id{MTLDevice}, Ptr{Cvoid}, Ptr{Cvoid}),
          dev, ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "respondsToSelector:"), sel) || return false
    return ccall(:objc_msgSend, Bool, (id{MTLDevice}, Ptr{Cvoid}), dev, sel)
end

"""
    MTL4CommandQueue(dev)

A Metal 4 submission queue.

Independent of the device's `MTLCommandQueue`s: a device may hold both, and work
on one is ordered against work on the other only through a shared event.
"""
function MTL4CommandQueue(dev::MTLDevice)
    q = @objc [dev::id{MTLDevice} newMTL4CommandQueue]::Union{Nothing,MTL4CommandQueue}
    q === nothing && throw(ArgumentError(
        "this device cannot create an MTL4CommandQueue; ask `supports_mtl4` first"))
    return q
end

"""
    MTL4CommandAllocator(dev)

The storage a command buffer records into — a command pool.

`reset!` hands all of it back at once, which is the only way an MTL4 command
buffer's memory is reclaimed. One allocator per frame-in-flight is the shape
this is for: reset it, refill the command buffer from it, submit, and do not
reset it again until the GPU has signalled that the submission landed.
"""
MTL4CommandAllocator(dev::MTLDevice) =
    @objc [dev::id{MTLDevice} newCommandAllocator]::MTL4CommandAllocator

"""Give the allocator's storage back. Only legal once the GPU is done with it."""
reset!(alloc::MTL4CommandAllocator) =
    @objc [alloc::id{MTL4CommandAllocator} reset]::Nothing

"""
    MTL4CommandBuffer(dev)

A REUSABLE command buffer.

Created once and refilled with [`begin_command_buffer!`](@ref) for every
submission, where the legacy path creates one per submission and throws it away.
"""
MTL4CommandBuffer(dev::MTLDevice) =
    @objc [dev::id{MTLDevice} newCommandBuffer]::MTL4CommandBuffer

"""Open the command buffer for recording, taking its storage from `alloc`."""
begin_command_buffer!(cb::MTL4CommandBuffer, alloc::MTL4CommandAllocator) =
    @objc [cb::id{MTL4CommandBuffer} beginCommandBufferWithAllocator:alloc::id{MTL4CommandAllocator}]::Nothing

"""Close the command buffer. It is submittable, and not yet submitted."""
end_command_buffer!(cb::MTL4CommandBuffer) =
    @objc [cb::id{MTL4CommandBuffer} endCommandBuffer]::Nothing

"""A compute encoder on an open MTL4 command buffer."""
compute_encoder(cb::MTL4CommandBuffer) =
    @objc [cb::id{MTL4CommandBuffer} computeCommandEncoder]::MTL4ComputeCommandEncoder

# The legacy method in `command_enc.jl` takes an `MTLCommandEncoderLike`; an MTL4
# encoder is a separate protocol, so it needs its own. Same name, because closing
# an encoder is one concept.
endEncoding!(enc::MTL4CommandEncoderLike) =
    @objc [enc::id{MTL4CommandEncoder} endEncoding]::Nothing

"""
Make a residency set's allocations resident for this command buffer.

The replacement for `useResource:`, and the only residency mechanism MTL4 has.
Declaring the set on the QUEUE ([`add_residency_set!`](@ref)) covers every
submission on it; declaring it here covers this one.
"""
use_residency_set!(cb::MTL4CommandBuffer, set::MTLResidencySet) =
    @objc [cb::id{MTL4CommandBuffer} useResidencySet:set::id{MTLResidencySet}]::Nothing

"""Make a residency set's allocations resident for everything on this queue."""
add_residency_set!(q::MTL4CommandQueue, set::MTLResidencySet) =
    @objc [q::id{MTL4CommandQueue} addResidencySet:set::id{MTLResidencySet}]::Nothing

"""
    commit!(queue, cmdbufs)

Submit closed command buffers, in order.

`commit:count:` takes an array because a queue submission is a batch here, the
way `vkQueueSubmit` takes one. The single-buffer method is the common case and
allocates the one-element array it needs.
"""
function commit!(q::MTL4CommandQueue, cbs::Vector{MTL4CommandBuffer})
    isempty(cbs) && return
    @objc [q::id{MTL4CommandQueue} commit:cbs::id{MTL4CommandBuffer}
                                   count:length(cbs)::Csize_t]::Nothing
end

# ── When a submission fails ──────────────────────────────────────────────────
#
# MTL4 has no `waitUntilCompleted` and no error property on a command buffer, so a
# submission that faults reports NOTHING: the event it would have signalled never
# advances, and a caller waiting on that value blocks forever with nothing in the
# system log to read. That is not a hypothetical — it is how a GPU fault in a
# replayed plan presents, and it costs hours to tell apart from "the GPU is busy".
#
# `commit:count:options:` with a feedback handler is the error path. The handler
# runs on a dispatch worker, so what it does here is the minimum that can be done
# there safely: read the error object out of the feedback and store its pointer.
# Reading the message is the WAITER's job, on a Julia thread.

"""What a queue's last failed submission reported, or `nothing`."""
mutable struct MTL4Feedback
    options::MTL4CommitOptions
    # The `id<NSError>` of the first failure, as an integer so the handler stores
    # a word and touches nothing that can allocate or yield.
    err::Base.RefValue{UInt}
    block::Any      # rooted: the block must outlive every submission using it
end

function MTL4Feedback()
    opts = @objc [MTL4CommitOptions alloc]::id{MTL4CommitOptions}
    opts = @objc [opts::id{MTL4CommitOptions} init]::MTL4CommitOptions
    err = Ref(UInt(0))
    # No allocation, no lock, no yield: one message send and one store.
    handler = function (feedback::Ptr{Cvoid})
        e = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), feedback,
                  ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "error"))
        e == C_NULL || (err[] = UInt(e))
        return nothing
    end
    block = @objcblock(handler, Nothing, (Ptr{Cvoid},))
    @objc [opts::id{MTL4CommitOptions} addFeedbackHandler:block::id{NSBlock}]::Nothing
    return MTL4Feedback(opts, err, block)
end

"""
    failure(fb) -> String or nothing

The message of the first submission that failed on this queue, if any.
"""
function failure(fb::MTL4Feedback)
    e = fb.err[]
    e == 0 && return nothing
    ns = reinterpret(id{NSError}, reinterpret(Ptr{Cvoid}, e))
    return String(NSString(@objc [ns::id{NSError} localizedDescription]::id{NSString}))
end

"""Submit with a feedback handler attached, so a failure is reportable."""
function commit!(q::MTL4CommandQueue, cbs::Vector{MTL4CommandBuffer}, fb::MTL4Feedback)
    isempty(cbs) && return
    @objc [q::id{MTL4CommandQueue} commit:cbs::id{MTL4CommandBuffer}
                                   count:length(cbs)::Csize_t
                                   options:fb.options::id{MTL4CommitOptions}]::Nothing
end

"""
    signal_event!(queue, event, value)

Signal `event` to `value` once everything committed so far has completed.

The whole of completion tracking on MTL4: there is no `waitUntilCompleted` and no
completion handler on a command buffer. `event.signaledValue >= v` is then a
non-blocking "has the GPU got this far", and
`MTL.waitUntilSignaledValue(event, v)` is the blocking form.
"""
signal_event!(q::MTL4CommandQueue, ev::MTLSharedEvent, value::Integer) =
    @objc [q::id{MTL4CommandQueue} signalEvent:ev::id{MTLEvent} value:UInt64(value)::UInt64]::Nothing

"""Make everything committed after this wait for `event` to reach `value`."""
wait_for_event!(q::MTL4CommandQueue, ev::MTLSharedEvent, value::Integer) =
    @objc [q::id{MTL4CommandQueue} waitForEvent:ev::id{MTLEvent} value:UInt64(value)::UInt64]::Nothing

# ── Arguments, which are addresses ───────────────────────────────────────────

"""
    MTL4ArgumentTable(dev; buffers, textures, samplers, initialize)

The binding table an MTL4 dispatch reads its arguments from.

There is no `setBytes`/`setBuffer` on the encoder: a shader's buffer argument at
index `i` is whatever 64-bit GPU address slot `i` of the bound table holds. That
is the same indirection a recorded command already uses, which is why a replay
needs no table at all — only an ordinary launch does.

`initialize` zeroes the slots at creation; without it an unwritten slot is
undefined rather than null.
"""
function MTL4ArgumentTable(dev::MTLDevice; buffers::Integer = 8, textures::Integer = 0,
                           samplers::Integer = 0, initialize::Bool = true)
    desc = @objc [MTL4ArgumentTableDescriptor alloc]::id{MTL4ArgumentTableDescriptor}
    desc = @objc [desc::id{MTL4ArgumentTableDescriptor} init]::MTL4ArgumentTableDescriptor
    desc.maxBufferBindCount = UInt64(buffers)
    textures > 0 && (desc.maxTextureBindCount = UInt64(textures))
    samplers > 0 && (desc.maxSamplerStateBindCount = UInt64(samplers))
    desc.initializeBindings = initialize
    err = Ref{id{NSError}}(nil)
    table = @objc [dev::id{MTLDevice} newArgumentTableWithDescriptor:desc::id{MTL4ArgumentTableDescriptor}
                                      error:err::Ptr{id{NSError}}]::Union{Nothing,MTL4ArgumentTable}
    table === nothing && throw_error(err[])
    return table
end

"""
    set_address!(table, address, index)

Bind a GPU address to a ONE-BASED buffer slot.

One-based to match `set_kernel_buffer!` on an indirect command, which is the
other place this package binds an argument by index; the selector underneath is
zero-based.
"""
set_address!(table::MTL4ArgumentTable, address::Integer, index::Integer) =
    @objc [table::id{MTL4ArgumentTable} setAddress:UInt64(address)::UInt64
                                        atIndex:Csize_t(index - 1)::Csize_t]::Nothing

set_argument_table!(enc::MTL4ComputeCommandEncoder, table::MTL4ArgumentTable) =
    @objc [enc::id{MTL4ComputeCommandEncoder} setArgumentTable:table::id{MTL4ArgumentTable}]::Nothing

set_function!(enc::MTL4ComputeCommandEncoder, pipeline::MTLComputePipelineState) =
    @objc [enc::id{MTL4ComputeCommandEncoder} setComputePipelineState:pipeline::id{MTLComputePipelineState}]::Nothing

dispatch_threadgroups!(enc::MTL4ComputeCommandEncoder, groups::MTLSize, threads::MTLSize) =
    @objc [enc::id{MTL4ComputeCommandEncoder} dispatchThreadgroups:groups::MTLSize
                                              threadsPerThreadgroup:threads::MTLSize]::Nothing

# ── Hazards, which are the caller's ──────────────────────────────────────────

"""
    barrier!(enc; after = MTLStageDispatch, before = MTLStageDispatch,
             visibility = MTL4VisibilityOptionDevice)

Order work encoded before this point against work encoded after it.

MTL4 does no hazard tracking, so a write followed by a read of the same bytes is
a race unless this says otherwise. `after`/`before` name the pipeline stages
(`MTLStageDispatch` for compute, `MTLStageBlit` for copies,
`MTLStageAccelerationStructure` for a build), and `visibility` says how far the
write has to be pushed — `Device` for anything another dispatch will read.

Only the encoder-to-encoder form is wrapped: the queue-stage variants order
against other queues, which is a different question and has no caller here.
"""
barrier!(enc::MTL4CommandEncoderLike; after::MTLStages = MTLStageDispatch,
         before::MTLStages = MTLStageDispatch,
         visibility::MTL4VisibilityOptions = MTL4VisibilityOptionDevice) =
    @objc [enc::id{MTL4CommandEncoder} barrierAfterEncoderStages:after::MTLStages
                                       beforeEncoderStages:before::MTLStages
                                       visibilityOptions:visibility::MTL4VisibilityOptions]::Nothing

# ── Replaying a recording ────────────────────────────────────────────────────
#
# The same two calls as on a legacy encoder, and the reason a recorded plan
# migrates for free: an indirect command buffer is an MTL4 object as much as a
# legacy one, and the commands inside it are unchanged.

"""Replay a ONE-BASED range of an indirect command buffer's commands."""
function execute_commands!(enc::MTL4ComputeCommandEncoder, icb::MTLIndirectCommandBuffer,
                           range::UnitRange{<:Integer})
    r = NSRange(first(range) - 1, length(range))
    @objc [enc::id{MTL4ComputeCommandEncoder} executeCommandsInBuffer:icb::id{MTLIndirectCommandBuffer}
                                              withRange:r::NSRange]::Nothing
end

"""
Replay a range of commands the DEVICE decided — see the legacy method for what an
execution range is and why a recording needs one.

A DIFFERENT selector from the legacy path, and the same name only in this
wrapper: `executeCommandsInBuffer:indirectBuffer:` takes an `MTL4BufferRange`,
which is a GPU ADDRESS and a length, where the legacy form takes a buffer and an
offset. Sending the three-part legacy selector here is an unrecognised-selector
abort at the moment a gated plan first replays, and nothing earlier warns.

Eight bytes because an `MTLIndirectCommandBufferExecutionRange` is two `UInt32`,
a location and a length.
"""
function execute_commands_indirect!(enc::MTL4ComputeCommandEncoder,
                                    icb::MTLIndirectCommandBuffer,
                                    rangebuf::MTLBuffer, offset::Integer)
    r = MTL4BufferRange(MTLGPUAddress(UInt64(rangebuf.gpuAddress) + UInt64(offset)),
                        UInt64(8))
    @objc [enc::id{MTL4ComputeCommandEncoder} executeCommandsInBuffer:icb::id{MTLIndirectCommandBuffer}
                                              indirectBuffer:r::MTL4BufferRange]::Nothing
end
