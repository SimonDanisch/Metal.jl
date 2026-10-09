#
# command buffer descriptor
#

export MTLCommandBufferDescriptor

# @objcwrapper managed = true MTLCommandBufferDescriptor <: NSObject

function MTLCommandBufferDescriptor()
    return @objc [MTLCommandBufferDescriptor new]::MTLCommandBufferDescriptor
end



#
# command buffer
#

export MTLCommandBuffer, enqueue!, wait_scheduled, wait_completed, encode_signal!,
       encode_wait!, commit!, on_scheduled, on_completed, CommandBufferError

# @objcwrapper MTLCommandBuffer <: NSObject

function MTLCommandBuffer(queue::MTLCommandQueue)
    @objc [queue::id{MTLCommandQueue} commandBuffer]::MTLCommandBuffer
end

function MTLCommandBuffer(queue::MTLCommandQueue, desc::MTLCommandBufferDescriptor)
    @objc [queue::id{MTLCommandQueue} commandBufferWithDescriptor:desc::id{MTLCommandBufferDescriptor}]::MTLCommandBuffer
end

function MTLCommandBuffer(f::Base.Callable, queue::MTLCommandQueue,
                          desc::MTLCommandBufferDescriptor=MTLCommandBufferDescriptor())
    cmdbuf = MTLCommandBuffer(queue, desc)
    commit!(f, cmdbuf)
    return cmdbuf
end

@objcwrapper managed = false MTLCommandBufferRef <: MTLCommandBuffer

@doc """
    MTLCommandBufferRef

An `MTLCommandBuffer` whose reference count is kept BY HAND: an `isbits` wrapper
around the object pointer that never retains or releases anything on its own.

What it exists for is the command buffer a submission path opens per submission.
The managed `MTLCommandBuffer` is a mutable struct with a release finalizer, so
every one is a Julia heap allocation, and a renderer that submits one command
buffer a frame paid 32 bytes a frame for it, and 32 more for its encoder
(`MTLComputeCommandEncoderRef`). A frame loop that must not allocate cannot be
built on top of that. This wrapper is a pointer, stored inline wherever it is
stored, and costs nothing to create or to hand around.

The price is that Julia's GC no longer keeps the object alive, so the rule is
the Objective-C one, applied by every place that stores one: ONE `retain` per
stored reference, ONE `release` when that reference is dropped. A reference held
only for the duration of a call, while something else is known to hold one, needs
neither — which is what [`MTLCommandBufferRef(cmdbuf)`](@ref) is for.

Every method written against `MTLCommandBufferLike` takes one, and its properties
are the command buffer's: the wrapper is declared as a Julia subtype only so that
the Objective-C class hierarchy (and with it the property chain) is the same.
""" MTLCommandBufferRef

"""
    MTLCommandBufferRef(cmdbuf::MTLCommandBufferLike) -> MTLCommandBufferRef

The same command buffer, unmanaged. Takes NO reference: it is valid for as long as
`cmdbuf` keeps the object alive, and whoever stores it has to `retain` it.
"""
MTLCommandBufferRef(cmdbuf::MTLCommandBufferLike) =
    reinterpret(MTLCommandBufferRef, pointer(cmdbuf))
MTLCommandBufferRef(cmdbuf::MTLCommandBufferRef) = cmdbuf

"""
    MTLCommandBufferRef(queue::MTLCommandQueue) -> MTLCommandBufferRef

A NEW command buffer on `queue`, with one reference that belongs to the caller, who
gives it back with `release`. The unmanaged counterpart of `MTLCommandBuffer(queue)`.
"""
MTLCommandBufferRef(queue::MTLCommandQueue) =
    retain_autoreleased(MTLCommandBufferRef) do
        @objc [queue::id{MTLCommandQueue} commandBuffer]::id{MTLCommandBufferRef}
    end

"""
    retain_autoreleased(T, f) -> T

Call `f`, which returns an AUTORELEASED object (a `+0` result, which is what every
Metal factory method that is not `new...` returns), and keep it: the object
retained once more, as an unmanaged `T` whose reference belongs to the caller.

Inside an autorelease pool of its own, and that is not a formality. An autoreleased
object holds a second reference that only goes away when the innermost pool is
drained, and a thread with no pool open puts it in an implicit one that is never
drained. Measured on an M5: a command buffer `MTLCommandBuffer(queue)` made outside
any pool still had a retain count of 2 after it completed and every Julia reference
to it had been released — Mantle's replay opened one per frame that way, and its
encoder, and leaked both. Pushing and popping a pool is two thread-local pointer
operations and takes no lock, unlike `@autoreleasepool`; it is safe here because
nothing between the two can switch tasks, so the pop runs on the thread the push
did.
"""
@inline function retain_autoreleased(f::F, ::Type{T}) where {F,T}
    pool = ccall(:objc_autoreleasePoolPush, Ptr{Cvoid}, ())
    try
        obj = T(f())
        retain(obj)
        return obj
    finally
        ccall(:objc_autoreleasePoolPop, Cvoid, (Ptr{Cvoid},), pool)
    end
end

"""
    retained(cmdbuf::MTLCommandBufferLike) -> MTLCommandBuffer

A MANAGED wrapper of `cmdbuf` that holds a reference of its own, released by its
finalizer — for code that keeps a command buffer it was handed as an
`MTLCommandBufferRef` (a profiler collecting them, say) and does not count
references itself. Allocates, so not for a per-submission path.
"""
retained(cmdbuf::MTLCommandBufferRef) = retain(MTLCommandBuffer, pointer(cmdbuf))
retained(cmdbuf::MTLCommandBufferLike) = cmdbuf

"""Whether `cmdbuf` has run to the end, successfully or not."""
completed(cmdbuf::MTLCommandBufferLike) = cmdbuf.status >= MTLCommandBufferStatusCompleted

"""
    enqueue!(cmdbuf::MTLCommandBuffer)

Enqueueing a command buffer reserves a place for the command buffer on the command
queue without committing the command buffer for execution. When this command buffer
is later committed, it keeps its position in the queue. You enqueue command buffers
so that you can create multiple command buffers with a fixed order of execution without
encoding the command buffers serially. You can use other threads to encode commands
into the command buffers and those threads can complete in any order.

[enqueue](https://developer.apple.com/documentation/metal/mtlcommandbuffer/1443019-enqueue?language=objc)
"""
function enqueue!(cmdbuf::MTLCommandBufferLike)
    cmdbuf.status in (MTLCommandBufferStatusCompleted, MTLCommandBufferStatusEnqueued) &&
        error("Cannot enqueue an already enqueued command buffer")
    submit(cmdbuf) do
        @objc [cmdbuf::id{MTLCommandBuffer} enqueue]::Nothing
    end
end

const last_committed_lock = ReentrantLock()

struct CommandBufferErrorInfo
    domain::String
    code::Int
    description::String
    command_buffer_label::Union{Nothing,String}
    queue_label::Union{Nothing,String}
end

"""
    CommandBufferError

An error reported by one or more Metal command buffers at a synchronization boundary.
The `errors` field contains the original `NSError` domain, code, and description, plus
the command-buffer and queue labels when available.
"""
struct CommandBufferError <: Exception
    errors::Vector{CommandBufferErrorInfo}
end

function Base.showerror(io::IO, exc::CommandBufferError)
    n = length(exc.errors)
    print(io, "CommandBufferError: ", n == 1 ? "Metal command buffer failed" :
                                           "$n Metal command buffers failed")
    for info in exc.errors
        print(io, "\nNSError: ", info.description, " (", info.domain,
              ", code ", info.code, ")")
        labels = String[]
        info.command_buffer_label === nothing ||
            push!(labels, "command buffer $(repr(info.command_buffer_label))")
        info.queue_label === nothing || push!(labels, "queue $(repr(info.queue_label))")
        isempty(labels) || print(io, " [", join(labels, ", "), "]")
    end
end

"""
What one queue has committed and synchronization has not yet accounted for.

ONE record per queue for the life of the process, updated in place. It used to be
taken out of the table by every `synchronize` and made again by the next commit,
which is a struct, a vector and its memory per frame for a renderer that commits and
waits once a frame; and its entries were `MTLCommandBufferLike`, an abstract type,
so reading a status off one was a dynamic call that boxed the enum it returned.

The command buffers are `MTLCommandBufferRef`s, and each place one is stored here
holds a reference of its own: one per entry of `pending`, one for `last`. Without
those, an entry whose batch had already retired it would be a dangling pointer the
next prune reads a status from.
"""
mutable struct QueueSubmissionState
    # Committed and not yet seen to complete, in commit order.
    pending::Vector{MTLCommandBufferRef}
    # Diagnostics of failures that completed but no synchronization has reported yet.
    errors::Union{Nothing,Vector{CommandBufferErrorInfo}}
    # The newest commit: the queue runs in order, so waiting for it waits for all.
    last::Union{Nothing,MTLCommandBufferRef}
    # How many command buffers were ever recorded, and how many of those have been
    # pruned. A synchronization notes the first before it waits and checks the second
    # against it afterwards, which is what "everything committed before I started has
    # finished" means when other tasks keep committing in the meantime.
    committed::Int
    retired::Int
end

QueueSubmissionState() = QueueSubmissionState(MTLCommandBufferRef[], nothing, nothing, 0, 0)

const submission_state_per_queue = Dict{id{MTLCommandQueue},QueueSubmissionState}()

function command_buffer_error_info(cmdbuf::MTLCommandBufferLike)
    err = cmdbuf.error
    err === nothing && return CommandBufferErrorInfo(
        "MTLCommandBufferErrorDomain", 0, "Metal did not provide an NSError",
        _object_label(cmdbuf), _object_label(cmdbuf.commandQueue))
    return CommandBufferErrorInfo(String(err.domain), Int(err.code),
                                  String(err.localizedDescription),
                                  _object_label(cmdbuf),
                                  _object_label(cmdbuf.commandQueue))
end

function _object_label(obj)
    label = obj.label
    return label === nothing ? nothing : String(label)
end

# Keep this state-accounting primitive independent from Objective-C objects so its
# ordering and pruning semantics can be tested deterministically. `retire` is called
# on every pruned entry after its diagnostics have been read, which is where a
# reference-counted entry gives its reference back.
function _prune_completed_submissions!(pending, errors, is_completed, error_info,
                                       retire=Returns(nothing))
    n = 0
    for submission in pending
        is_completed(submission) || break
        info = error_info(submission)
        if info !== nothing
            if errors === nothing
                errors = [info]
            else
                push!(errors, info)
            end
        end
        retire(submission)
        n += 1
    end
    n == 0 || deleteat!(pending, 1:n)
    return errors
end

failure_info(cmdbuf::MTLCommandBufferRef) =
    cmdbuf.status == MTLCommandBufferStatusError ? command_buffer_error_info(cmdbuf) :
                                                   nothing

# `last_committed_lock` held.
function prune_completed_submissions!(state::QueueSubmissionState)
    before = length(state.pending)
    state.errors = _prune_completed_submissions!(state.pending, state.errors,
                                                 completed, failure_info, release)
    state.retired += before - length(state.pending)
    return
end

function record_committed!(cmdbuf::MTLCommandBufferLike, key::id{MTLCommandQueue})
    ref = MTLCommandBufferRef(cmdbuf)
    @lock last_committed_lock begin
        state = get!(QueueSubmissionState, submission_state_per_queue, key)
        # Completed successes can be forgotten immediately. Completed failures are
        # reduced to diagnostics, so command-buffer retention stays bounded during
        # long-running submission workloads without explicit synchronization.
        prune_completed_submissions!(state)
        # Two references, because it is stored twice: `pending` gives its own back
        # when the prune reaches it, `last` when the next commit replaces it.
        retain(ref)
        push!(state.pending, ref)
        retain(ref)
        previous = state.last
        state.last = ref
        previous === nothing || release(previous)
        state.committed += 1
    end
    return
end

"""
What a synchronization waits for on one queue: the newest command buffer committed
to it, or `nothing`, and how many command buffers had been committed to it by then.

A struct and not a tuple, and that is the whole reason it exists: a tuple holding a
`Union` is an abstract type, so returning `(last, committed)` boxed the tuple and both
of its elements — 56 bytes on every `synchronize`. A struct with a `Union` field is
concrete, and comes back without a heap allocation.
"""
struct SyncPoint
    queue::id{MTLCommandQueue}
    last::Union{Nothing,MTLCommandBufferRef}
    committed::Int
end

"""
    sync_point(queue) -> SyncPoint

Where a synchronization of `queue` has to get to.

`last` comes with a reference of its own that belongs to the CALLER, who releases it
once the wait is over. It has to: the wait happens without any lock held, and a
commit on another task replaces `last` meanwhile and gives back the table's
reference, which may have been the only one left.
"""
function sync_point(queue::MTLCommandQueue)
    key = pointer(queue)
    @lock last_committed_lock begin
        state = get(submission_state_per_queue, key, nothing)
        state === nothing && return SyncPoint(key, nothing, 0)
        return sync_point(key, state)
    end
end

# `last_committed_lock` held.
function sync_point(key::id{MTLCommandQueue}, state::QueueSubmissionState)
    last = state.last
    last === nothing || retain(last)
    return SyncPoint(key, last, state.committed)
end

"""
    sync_points() -> Vector{SyncPoint}

`sync_point` of every queue that has committed anything, for a device-wide
synchronization. Each `last` is the caller's to release.
"""
function sync_points()
    @lock last_committed_lock begin
        return SyncPoint[sync_point(key, state) for (key, state) in submission_state_per_queue]
    end
end

"""
    finish_submissions!(point::SyncPoint) -> errors

Account for a synchronization that has waited for `point`, and claim the failures
nobody has reported yet on its queue: the diagnostics, or `nothing`. The queue runs in
order, so after that wait every command buffer the point covers has to be done; one
that is not is an error here rather than a silent gap in the error report.
"""
function finish_submissions!(point::SyncPoint)
    @lock last_committed_lock begin
        state = get(submission_state_per_queue, point.queue, nothing)
        state === nothing && return nothing
        prune_completed_submissions!(state)
        state.retired >= point.committed ||
            error("Command buffer did not reach a terminal state after queue synchronization")
        errors = state.errors
        state.errors = nothing
        return errors
    end
end

"""
    all_completed(queue) -> Bool

Whether every command buffer committed to `queue` so far has completed — what
`last_committed(queue).status` answers, without making a wrapper to read it from.
"""
function all_completed(queue::MTLCommandQueue)
    @lock last_committed_lock begin
        state = get(submission_state_per_queue, pointer(queue), nothing)
        state === nothing && return true
        last = state.last
        return last === nothing || completed(last)
    end
end

function pending_submission_count(queue::MTLCommandQueue)
    key = pointer(queue)
    @lock last_committed_lock begin
        state = get(submission_state_per_queue, key, nothing)
        state === nothing ? 0 : length(state.pending)
    end
end

# optional profiling hook. when set, it is invoked for every committed command buffer.
const profile_hook = Ref{Any}(nothing)

# optional submission hook. when set, it is invoked as `hook(f, cmdbuf)` to enqueue or
# commit a command buffer, which happens by calling `f()`.
const submit_hook = Ref{Any}(nothing)

@inline function submit(f, cmdbuf::MTLCommandBufferLike)
    hook = submit_hook[]
    hook === nothing ? f() : hook(f, cmdbuf)
end

# optional profiling data for operation metadata (e.g. kernel dimensions, copy sizes).
const profile_metadata = Ref{Any}(nothing)

struct ProfileCollector
    lock::ReentrantLock
    # Keyed by the command buffer's ADDRESS, not by the wrapper: a batch's command
    # buffer is an `MTLCommandBufferRef` when its operations are noted and may be
    # another wrapper of the same object by the time it is read. Every command buffer
    # noted here is committed right after and kept alive by `records`, so an address
    # is not reused while the collector holds it.
    metadata::Dict{UInt,Vector{Any}}
    records::Vector{Tuple{String,Any}}
end

ProfileCollector() = ProfileCollector(ReentrantLock(), Dict{UInt,Vector{Any}}(),
                                      Tuple{String,Any}[])

@inline function note_operation!(collector::ProfileCollector, cmdbuf::MTLCommandBufferLike, op)
    @lock collector.lock begin
        ops = get!(Vector{Any}, collector.metadata, UInt(pointer(cmdbuf)))
        push!(ops, op)
    end
    return
end

"""
    last_committed(queue::MTLCommandQueue)::Union{MTLCommandBuffer, Nothing}

Return the most recently committed command buffer on `queue`, or `nothing` if
nothing has been committed. A managed wrapper with a reference of its own, so it
allocates; [`all_completed`](@ref) asks the common question without one.
"""
function last_committed(queue::MTLCommandQueue)
    @lock last_committed_lock begin
        state = get(submission_state_per_queue, pointer(queue), nothing)
        state === nothing && return nothing
        last = state.last
        return last === nothing ? nothing : retained(last)
    end
end

function commit!(cmdbuf::MTLCommandBufferLike)
    submit(cmdbuf) do
        commit_with_queue_key!(cmdbuf, pointer(cmdbuf.commandQueue))
    end
end

function commit!(cmdbuf::MTLCommandBufferLike, queue::MTLCommandQueue)
    submit(cmdbuf) do
        commit_with_queue_key!(cmdbuf, pointer(queue))
    end
end

function commit_with_queue_key!(cmdbuf::MTLCommandBufferLike, key::id{MTLCommandQueue})
    cmdbuf.status in (MTLCommandBufferStatusCompleted, MTLCommandBufferStatusCommitted) &&
        error("Cannot commit an already committed/completed command buffer")
    @objc [cmdbuf::id{MTLCommandBuffer} commit]::Nothing
    # Record every submission for error accounting. The most recent buffer remains
    # the queue tail used by synchronization, while older completed buffers are
    # pruned or summarized by `record_committed!`.
    record_committed!(cmdbuf, key)
    hook = profile_hook[]
    hook === nothing || hook(cmdbuf)
    return
end

function commit!(f::Base.Callable, cmdbuf::MTLCommandBufferLike)
    enqueue!(cmdbuf)
    ret = f(cmdbuf)
    commit!(cmdbuf)
    return ret
end

function use_residency_set!(cmdbuf::MTLCommandBufferLike, resset::MTLResidencySet)
    @objc [cmdbuf::id{MTLCommandBuffer} useResidencySet:resset::id{MTLResidencySet}]::Nothing
end

function use_residency_sets!(cmdbuf::MTLCommandBufferLike, ressets, count)
    @objc [cmdbuf::id{MTLCommandBuffer} useResidencySets:ressets::Ptr{id{MTLResidencySet}}
                                                count:count::NSUInteger]::Nothing
end

"""
    wait_scheduled(commandBuffer)

Blocks execution of the current thread until the command buffer is scheduled.
This method returns after the command buffer has been scheduled and all code
blocks registered by addScheduledHandler: have been invoked. A command buffer
is considered scheduled after all its dependencies are resolved, and it is sent
to the GPU for execution.
"""
function wait_scheduled(cmdbuf::MTLCommandBufferLike)
    @objc [cmdbuf::id{MTLCommandBuffer} waitUntilScheduled]::Nothing
end

"""
    wait_completed(cmdbuf::MTLCommandBuffer)

Blocks execution of the current thread until execution of the command
buffer is completed.
This method returns after the command buffer is completed and all code
blocks registered by addCompletedHandler: are invoked.
"""
function wait_completed(cmdbuf::MTLCommandBufferLike)
    @objc [cmdbuf::id{MTLCommandBuffer} waitUntilCompleted]::Nothing
end

"""
    encode_signal!(cmdbuf::MTLCommandBuffer, ev::MTLEvent, val::UInt)

Encodes a command that signals the given event, updating it to a new value.

You can't encode a signal event if the command buffer has an active command encoder.
Metal signals the event after all commands scheduled prior to this command
have finished executing. If the new event value is greater than the event's
current value, Metal updates the event's value to the new value. Commands
waiting on the event are allowed to run if the new value is equal to or
greater than the value for which they are waiting. For shared events, this
update similarly triggers notification handlers waiting on the event.
"""
function encode_signal!(cmdbuf::MTLCommandBufferLike,
                         ev, val::Integer)
    @objc [cmdbuf::id{MTLCommandBuffer} encodeSignalEvent:ev::id{MTLEvent}
                                     value:val::UInt64]::Nothing
end

"""
    encode_wait!(cmdbuf::MTLCommandBuffer, ev::MTLEvent, val::UInt)

Encodes a command that blocks the execution of the command buffer
until the given event reaches the given value.

You can't encode a signal event if the command buffer has an active command encoder.
When the device object reaches the command for the wait event, the
device object waits until the event is signaled with a value
equal to or larger than the provided value. While waiting, the
GPU executes commands that appear earlier than the wait command,
but doesn't start any commands that appear after it. Execution continues
immediately if the event already has an equal or larger value.
"""
function encode_wait!(cmdbuf::MTLCommandBufferLike,
                       ev, val::Integer)
    @objc [cmdbuf::id{MTLCommandBuffer} encodeWaitForEvent:ev::id{MTLEvent}
                                     value:val::UInt64]::Nothing
end

function _command_buffer_callback(f)
    # convert the incoming pointer, and discard any return value
    function wrapper(ptr)
        try
            f(ptr == nil ? nothing : MTLCommandBuffer(ptr))
        catch err
            # we might be on an unmanaged thread here, so display the error
            # (otherwise it may get lost, or worse, crash Julia)
            @error "Command buffer callback encountered an error: " * sprint(showerror, err)
        end
        return
    end
    @objcblock(wrapper, Nothing, (id{MTLCommandBuffer},))
end

"""
    on_scheduled(cmdbuf::MTLCommandBuffer) do cbuf
        ...
        return
    end

Execute a block of code when execution of the command buffer is scheduled.
"""
function on_scheduled(f::Base.Callable, cmdbuf::MTLCommandBufferLike)
    block = _command_buffer_callback(f)
    @objc [cmdbuf::id{MTLCommandBuffer} addScheduledHandler:block::id{NSBlock}]::Nothing
end

"""
    on_completed(cmdbuf::MTLCommandBuffer) do cbuf
        ...
        return
    end

Execute a block of code when execution of the command buffer is completed.
"""
function on_completed(f::Base.Callable, cmdbuf::MTLCommandBufferLike)
    block = _command_buffer_callback(f)
    @objc [cmdbuf::id{MTLCommandBuffer} addCompletedHandler:block::id{NSBlock}]::Nothing
end

"""
    on_completed(cmdbuf::MTLCommandBuffer, cond::Base.AsyncCondition)

Signal `cond` when execution of the command buffer is completed, without running
Julia code on Metal's completion-handler thread.
"""
function on_completed(cmdbuf::MTLCommandBufferLike, cond::Base.AsyncCondition)
    block = @objcasyncblock(cond)
    @objc [cmdbuf::id{MTLCommandBuffer} addCompletedHandler:block::id{NSBlock}]::Nothing
end
