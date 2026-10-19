export MTL4ComputeCommandEncoderRef, MTL4Feedback, dispatch_threadgroups_indirect!

# Metal 4 submission, which is the model Metal has been moving towards and the one
# a render graph already wanted.
#
# `libmtl.jl` (generated) has the TYPES; `command_queue4.jl`, `command_alloc4.jl`,
# `command_buf4.jl`, `command_enc4.jl`, `command_enc/compute4.jl` and `arg_table.jl`
# have the methods upstream wraps. This file is what they lack: an unmanaged encoder
# for a per-submission path, a feedback handler that runs no Julia code that can
# allocate, a device-decided grid, and replaying an indirect command buffer.
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
# Everything here is macOS 26 and later. `Metal.metal4_unsupported_reason` is the
# per-device question.

@objcwrapper managed = false MTL4ComputeCommandEncoderRef <: MTL4ComputeCommandEncoder

@doc """
    MTL4ComputeCommandEncoderRef(cb::MTL4CommandBuffer)

A NEW compute encoder on `cb`, as an unmanaged wrapper whose one reference belongs to
the caller; `close` ends it and gives the reference back.

What a submission path opens once per submission, for the reason
[`MTLCommandBufferRef`](@ref) gives: the managed `MTL4ComputeCommandEncoder(cb)` is a
heap object per call, and it was made outside any autorelease pool, so the encoder's
autoreleased reference was never given back either.
""" MTL4ComputeCommandEncoderRef

MTL4ComputeCommandEncoderRef(cb::MTL4CommandBuffer) =
    retain_autoreleased(MTL4ComputeCommandEncoderRef) do
        @objc [cb::id{MTL4CommandBuffer} computeCommandEncoder]::id{MTL4ComputeCommandEncoderRef}
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
    opts = MTL4CommitOptions()
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
commit!(q::MTL4CommandQueue, cbs::Vector{MTL4CommandBuffer}, fb::MTL4Feedback) =
    isempty(cbs) ? nothing : commit!(q, cbs, fb.options)

"""
    commit!(queue, cbs::Vector{id{MTL4CommandBuffer}}, fb)

The same, over an ALREADY-MARSHALLED array of command buffer pointers. The method above
maps its `Vector{MTL4CommandBuffer}` to a fresh pointer array on every call, so a caller
that keeps its array to submit without allocating still allocated there. The caller keeps the command
buffers alive; these are bare pointers.
"""
function commit!(q::MTL4CommandQueue, cbs::Vector{id{MTL4CommandBuffer}}, fb::MTL4Feedback)
    isempty(cbs) && return
    @objc [q::id{MTL4CommandQueue} commit:cbs::Ptr{id{MTL4CommandBuffer}}
                                   count:length(cbs)::Csize_t
                                   options:fb.options::id{MTL4CommitOptions}]::Nothing
end

"""
    dispatch_threadgroups_indirect!(enc, buf, offset, threads)

Dispatch a grid the DEVICE decided: `buf` holds three `UInt32` at `offset`, the
threadgroup counts, and zero in the first runs nothing.

This is how a gate survives without an indirect command buffer. A recorded plan's
`repeat!` gate is a zero-LENGTH execution range replayed from one; encoded
directly it is a zero-COUNT grid instead, written by the same one-thread kernel,
and the host still never learns whether the iteration ran.
"""
function dispatch_threadgroups_indirect!(enc::MTL4ComputeCommandEncoderLike, buf::MTLBuffer,
                                         offset::Integer, threads::MTLSize)
    a = MTLGPUAddress(UInt64(gpu_address(buf)) + UInt64(offset))
    @objc [enc::id{MTL4ComputeCommandEncoder} dispatchThreadgroupsWithIndirectBuffer:a::MTLGPUAddress
                                              threadsPerThreadgroup:threads::MTLSize]::Nothing
end

# ── Replaying a recording ────────────────────────────────────────────────────
#
# The same two calls as on a legacy encoder, and the reason a recorded plan
# migrates for free: an indirect command buffer is an MTL4 object as much as a
# legacy one, and the commands inside it are unchanged.

"""Replay a ONE-BASED range of an indirect command buffer's commands."""
function execute_commands!(enc::MTL4ComputeCommandEncoderLike, icb::MTLIndirectCommandBuffer,
                           range::UnitRange{<:Integer})
    r = NSRange(first(range) - 1, length(range))
    @objc [enc::id{MTL4ComputeCommandEncoder} executeCommandsInBuffer:icb::id{MTLIndirectCommandBuffer}
                                              withRange:r::NSRange]::Nothing
end

"""
Replay a range of commands the DEVICE decided — see the legacy method for what an
execution range is and why a recording needs one.

A DIFFERENT selector from the legacy path, and the same name only in this
wrapper: `executeCommandsInBuffer:indirectBuffer:` takes a bare `MTLGPUAddress`,
where the legacy form takes a buffer and an offset. Sending the three-part legacy
selector here is an unrecognised-selector abort at the moment a gated plan first
replays, and nothing earlier warns.

No length goes with the address, unlike everything else MTL4 addresses by
`MTL4BufferRange`: the command processor reads the two `UInt32` of an
`MTLIndirectCommandBufferExecutionRange` and that is the whole of it.

Correct, and unusable for what it is for: this call stops completing after about a
second of cumulative replayed GPU work, with no error and nothing in the system
log. `research/mtl4_icb_stall.jl` reproduces it and rules the causes out one knob
at a time; `dispatch_threadgroups_indirect!` is what a device-decided grid has to
go through instead.
"""
function execute_commands_indirect!(enc::MTL4ComputeCommandEncoderLike,
                                    icb::MTLIndirectCommandBuffer,
                                    rangebuf::MTLBuffer, offset::Integer)
    a = MTLGPUAddress(UInt64(gpu_address(rangebuf)) + UInt64(offset))
    @objc [enc::id{MTL4ComputeCommandEncoder} executeCommandsInBuffer:icb::id{MTLIndirectCommandBuffer}
                                              indirectBuffer:a::MTLGPUAddress]::Nothing
end
