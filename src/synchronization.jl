export synchronize, device_synchronize, CommandBufferError

# whether to wait without blocking the calling thread, instead of parking it inside Metal's
# blocking `waitUntilCompleted`. opt-out via Preferences for bisection or to compare
# against the blocking baseline.
const use_nonblocking_synchronization =
    @load_preference("nonblocking_synchronization", true)

is_completed(cmdbuf::MTL.MTLCommandBufferLike) =
    cmdbuf.status >= MTL.MTLCommandBufferStatusCompleted

# blocking wait, performed on a waiter's thread. `@objc` calls are GC-safe, but the thread
# has no autorelease pool of its own, so set one up. it cannot be an `@autoreleasepool`,
# whose global lock may be held by the waiting task.
function blocking_wait(cmdbuf::MTL.MTLCommandBufferLike)
    pool = ccall(:objc_autoreleasePoolPush, Ptr{Cvoid}, ())
    try
        wait_completed(cmdbuf)
    finally
        ccall(:objc_autoreleasePoolPop, Cvoid, (Ptr{Cvoid},), pool)
    end
    return
end

# wait for a committed command buffer to complete, without blocking the calling thread so
# that other tasks can run in the meantime. pass `handlers=true` to also wait for its
# completion handlers to have run (e.g., to flush `addLogHandler:` output); otherwise, this
# may or may not return before they have run.
#
# note that long waits use `waitUntilCompleted`, which waits for the completion handlers, so
# these must not depend on the waiting task, e.g., on locks it holds (like the global lock
# taken by `@autoreleasepool`).
function wait_cmdbuf!(cmdbuf::MTL.MTLCommandBufferLike; handlers::Bool=false)
    !handlers && is_completed(cmdbuf) && return

    if !use_nonblocking_synchronization
        wait_completed(cmdbuf)
    elseif GC.in_finalizer() || ccall(:jl_generating_output, Cint, ()) != 0
        # no switching tasks here, so no handing the wait to another thread either
        blocking_wait(cmdbuf)
    else
        # A view, not a reference: the caller's own keeps `cmdbuf` alive for the call,
        # and the waiter takes one of its own for its thread.
        ref = MTL.MTLCommandBufferRef(cmdbuf)
        (handlers || !poll_completed(ref)) && wait_on_waiter!(ref, !handlers)
    end
    return
end

# Short waits are the common case, and handing one to another thread costs a couple of
# microseconds, more when that thread has gone to sleep: poll first, busy at the start and
# then yielding to other tasks. The same budget as GPUToolbox's `cooperative_wait`, whose
# polling this replaces along with its slow path (see `CommandBufferWaiter`). Measured on an
# M5 in a process with nothing else to run, the budget runs out after about 2.6 ms, most of
# it in the yields; a one-dispatch frame waits about 190 us and stayed inside it in 5000
# waits out of 5000, and a frame with more GPU work than the budget never does.
function poll_completed(cmdbuf::MTL.MTLCommandBufferRef; busy::Int=32, total::Int=256)
    for i in 1:total
        if i <= busy
            ccall(:jl_cpu_pause, Cvoid, ())
            GC.safepoint()
        else
            yield()
        end
        is_completed(cmdbuf) && return true
    end
    return false
end

"""
A thread of Metal.jl's own that blocks in `waitUntilCompleted` on behalf of a waiting
task, so that the task's thread can run other tasks until the command buffer is done.

This is the design of GPUToolbox's `cooperative_wait`, which `wait_cmdbuf!` used before,
with one difference, and it is the reason this exists: NOTHING here is made per wait.
`cooperative_wait` builds a `WaitState`, a `WaitRequest` and the `Base.Event` inside it —
its lock, its condition and two wait lists — for every wait that outlasts its polling,
and boxes the object it waits on into an `Any` field: 272 bytes, measured, for one wait.
Every frame whose GPU work outlasts the polling (about 2.6 ms on an M5) waits that long,
so a renderer allocated those bytes once a frame for nothing it kept: a Mantle plan of one
18 ms dispatch measured 280 bytes per `run!`/`waitfor!` cycle that way, and 0 with this.
Here a waiter, its two events and its thread are made once and used again, and the command
buffer is a field of its own type, so handing it over boxes nothing.

Handing over is the waiting task setting `cmdbuf` and notifying `work`; the thread waits
for the command buffer, gives back the reference it was handed, and notifies `done`. The
waiter goes back to the pool when its TASK has consumed `done`, not when its thread is
finished: a thread that returned itself first could be handed the next wait while the
previous task had not yet woken, and that task would consume the next wait's `done`.

One thread per waiter, created when a wait finds none idle, and kept. At most four,
as GPUToolbox has, because a driver may spin while it waits and every busy waiter would
then hold a core: a wait that finds four busy polls its command buffer instead and takes
the first waiter that comes free. Twelve tasks on four threads, each waiting on a long
command buffer eight times, made twelve threads before the cap. A wait that has to see the
completion handlers run (`handlers = true`) cannot be polled, so it makes a waiter past
the cap rather than wait for one.
"""
mutable struct CommandBufferWaiter
    const work::Base.Event
    const done::Base.Event
    cmdbuf::Union{Nothing,MTL.MTLCommandBufferRef}
    # What the wait threw on the waiter's thread, for the waiting task to throw.
    failure::Any
end

CommandBufferWaiter() = CommandBufferWaiter(Base.Event(true), Base.Event(true), nothing, nothing)

# every waiter ever made, keeping them rooted (their threads hold only a pointer), and the
# ones not handling a wait
const waiters = CommandBufferWaiter[]
const idle_waiters = CommandBufferWaiter[]
const waiters_lock = ReentrantLock()

function waiter_loop(data::Ptr{Cvoid})
    w = unsafe_pointer_to_objref(data)::CommandBufferWaiter
    # never run finalizers on this thread: one that blocks (freeing GPU memory can wait for
    # the device) would keep it from notifying the task that is waiting on it.
    ccall(:jl_gc_disable_finalizers_internal, Cvoid, ())
    while true
        wait(w.work)
        cmdbuf = w.cmdbuf::MTL.MTLCommandBufferRef
        try
            blocking_wait(cmdbuf)
        catch err
            # passed on, and thrown by the waiting task
            w.failure = err
        finally
            # the reference the waiting task took for this thread
            release(cmdbuf)
        end
        notify(w.done)
    end
end

function start_waiter!(w::CommandBufferWaiter)
    # the size of a `uv_thread_t` is not known here, so reserve enough
    tid = Ref{NTuple{32,UInt8}}(ntuple(_ -> 0x00, 32))
    cb = @cfunction(waiter_loop, Cvoid, (Ptr{Cvoid},))
    err = ccall(:uv_thread_create, Cint, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                tid, cb, pointer_from_objref(w))
    err == 0 || Base.uv_error("uv_thread_create", err)
    ccall(:uv_thread_detach, Cint, (Ptr{Cvoid},), tid)
    return w
end

# An idle waiter, a new one, or `nothing` when `limit` are busy and the wait can be polled.
# A new waiter is counted under the lock and started after it, so that two waits deciding at
# once cannot both make the fifth.
function take_waiter!(pollable::Bool; limit::Int=4)
    w = Base.@lock waiters_lock begin
        isempty(idle_waiters) || return pop!(idle_waiters)
        pollable && length(waiters) >= limit && return nothing
        fresh = CommandBufferWaiter()
        push!(waiters, fresh)
        fresh
    end
    return start_waiter!(w)
end

# Hand the wait for `cmdbuf` to a waiter and wait for it to say the command buffer is done.
#
# Interrupted, this keeps waiting and throws the interrupt afterwards, as
# `cooperative_wait` does: the command buffer may still be using memory the caller would
# release when unwinding. Any other exception thrown into the task is thrown at once; the
# waiter is then not returned to the pool, because its `done` will be set by a wait nobody
# consumes, and its thread still holds its own reference to the command buffer, so
# nothing it reads is released under it.
function wait_on_waiter!(cmdbuf::MTL.MTLCommandBufferRef, pollable::Bool)
    w = take_waiter!(pollable)
    while w === nothing
        # every waiter is busy: poll, and look for a free one every 100 us
        t0 = time_ns()
        while time_ns() - t0 < 100_000
            is_completed(cmdbuf) && return
            yield()
        end
        w = take_waiter!(pollable)
    end
    retain(cmdbuf)
    w.cmdbuf = cmdbuf
    notify(w.work)

    interrupt = nothing
    while true
        try
            wait(w.done)
            break
        catch err
            err isa InterruptException || rethrow()
            interrupt = err
        end
    end

    w.cmdbuf = nothing
    failure = w.failure
    w.failure = nothing
    Base.@lock waiters_lock push!(idle_waiters, w)
    failure === nothing || throw(failure)
    interrupt === nothing || throw(interrupt)
    return
end

# The failures collected so far, and the ones one more queue reported.
merge_errors(::Nothing, more) = more
merge_errors(errors::Vector{MTL.CommandBufferErrorInfo}, ::Nothing) = errors
merge_errors(errors::Vector{MTL.CommandBufferErrorInfo},
             more::Vector{MTL.CommandBufferErrorInfo}) = append!(errors, more)

function check_synchronization_errors(errors::Union{Nothing,Vector{MTL.CommandBufferErrorInfo}})
    kernel_error = try
        check_exceptions()
        nothing
    catch err
        err
    end

    command_buffer_error = errors === nothing ? nothing : CommandBufferError(errors)
    if command_buffer_error !== nothing && kernel_error !== nothing
        throw(CompositeException(Any[command_buffer_error, kernel_error]))
    elseif command_buffer_error !== nothing
        throw(command_buffer_error)
    elseif kernel_error !== nothing
        throw(kernel_error)
    end
    return
end

# Wait for `cmdbuf`, whose reference is the caller's (`MTL.sync_point` hands one out
# with it), and give that reference back however the wait ends.
wait_and_release!(::Nothing) = nothing
function wait_and_release!(cmdbuf::MTL.MTLCommandBufferRef)
    try
        wait_cmdbuf!(cmdbuf)
    finally
        release(cmdbuf)
    end
    return
end


#
# public API
#

"""
    synchronize(queue=global_queue(device()))

Wait for currently committed GPU work on `queue` to finish. This includes work left
behind by tasks that have finished, so that their results are visible after `wait`ing
for them.
"""
synchronize() = synchronize(global_queue(device()))

synchronize(queue::MTLCommandQueue) = synchronize(@autoreleasepool batched_queue(queue))

# A renderer calls this once a frame, through Mantle's `waitfor!`, so the steady state
# allocates NOTHING, and the shape below is what that took. The two pools are
# functions of their own: as `@autoreleasepool begin … end` blocks inside this body,
# the pool's closure captured a variable that was assigned again after it (two
# `Core.Box`es a call), and handed back a tuple of a queue and two `Union`s, which was
# boxed too. The wait between them holds no pool, because `@autoreleasepool` takes a
# global lock that other tasks would then be waiting on.
function synchronize(bq::BatchedCommandQueue)
    # A scan, and in the steady state an empty one: see `orphaned_batched_queues`.
    orphans = orphaned_batched_queues()
    upto = commit_for_synchronize!(bq, orphans)
    queue = bq.queue

    # flush any pending log handlers from logging-enabled kernels on this queue
    # (Metal delivers logs asynchronously; waiting for the specific cmdbuf's
    # completion handlers is what processes its `addLogHandler:` blocks)
    drain_logging_cmdbufs!(queue)

    # Handles the already-completed fast path internally.
    point = MTL.sync_point(queue)
    wait_and_release!(point.last)

    points = orphans === nothing ? nothing : wait_orphaned_queues!(orphans)
    finish_synchronize!(bq, upto, point, orphans, points)
    return
end

# Commit what is open, on `bq` and on the queues finished tasks left behind, and say
# how many command buffers `bq` had then handed to `defer_cleanup!`: those are the
# ones this synchronization retires.
@autoreleasepool function commit_for_synchronize!(bq::BatchedCommandQueue, orphans)
    upto = Base.@lock submission_lock begin
        commit_batch!(bq)
        bq.ncommitted
    end
    # their owner will not touch them again, so we can, but other tasks synchronizing
    # may do so concurrently.
    orphans === nothing || foreach(commit_batch!, orphans)
    maybe_collect(bq.device; will_block=true)
    return upto
end

# Release what the waited-for work held, then report how it went.
@autoreleasepool function finish_synchronize!(bq::BatchedCommandQueue, upto::Int,
                                              point::MTL.SyncPoint, orphans, points)
    # other tasks may have committed more work to this queue while we were waiting,
    # so only force the cleanup of command buffers that were committed before.
    drain_cleanups!(bq; until=upto)
    errors = MTL.finish_submissions!(point)
    if orphans !== nothing
        foreach(drain_cleanups!, orphans)
        for p in points
            errors = merge_errors(errors, MTL.finish_submissions!(p))
        end
    end

    # Surface Metal runtime failures and device-side Julia exceptions together,
    # after cleanup has released all Julia roots held by completed work.
    check_synchronization_errors(errors)
    return
end

# wait for the work committed to `bq`, which may belong to another task. errors are left
# for the queue's owner to report when it synchronizes, unless it has finished.
function synchronize_queue(bq::BatchedCommandQueue)
    (bq.owner === current_task() || istaskdone(bq.owner)) && return synchronize(bq)

    @autoreleasepool Base.@lock submission_lock commit_batch!(bq)
    wait_and_release!(MTL.sync_point(bq.queue).last)
    return
end

# wait for the queues finished tasks left work in, and say where each of them got to
function wait_orphaned_queues!(orphans::Vector{BatchedCommandQueue})
    points = MTL.SyncPoint[]
    for bq in orphans
        drain_logging_cmdbufs!(bq.queue)
        point = MTL.sync_point(bq.queue)
        wait_and_release!(point.last)
        push!(points, point)
    end
    return points
end

"""
    synchronize(cmdbuf::MTLCommandBufferLike)

Wait for `cmdbuf` (which must already have been committed) and all preceding
work on the same queue to complete.
"""
synchronize(cmdbuf::MTL.MTLCommandBufferLike) = synchronize(cmdbuf.commandQueue)

"""
    device_synchronize()

Synchronize all committed GPU work across all global queues.
"""
function device_synchronize()
    flush_batched_queues!()
    maybe_collect(device(); will_block=true)

    queues = active_global_queues()
    append!(queues, active_batched_queues())
    for queue in unique!(queues)
        drain_logging_cmdbufs!(raw_queue(queue))
    end

    # the last command buffer committed to each queue completes after the earlier ones.
    # every `last` comes with a reference of ours, given back however the waits end.
    points = MTL.sync_points()
    try
        for p in points
            p.last === nothing || wait_cmdbuf!(p.last)
        end
    finally
        for p in points
            p.last === nothing || release(p.last)
        end
    end

    # other tasks may have committed work while we were waiting, so only clean up after
    # command buffers that have completed
    for bq in active_batched_queues()
        drain_cleanups!(bq)
    end

    errors = nothing
    for p in points
        errors = merge_errors(errors, MTL.finish_submissions!(p))
    end
    check_synchronization_errors(errors)
    return
end
