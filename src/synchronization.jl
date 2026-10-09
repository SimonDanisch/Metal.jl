export synchronize, device_synchronize, CommandBufferError

# whether to wait without blocking the calling thread, instead of parking it inside Metal's
# blocking `waitUntilCompleted`. opt-out via Preferences for bisection or to compare
# against the blocking baseline.
const use_nonblocking_synchronization =
    @load_preference("nonblocking_synchronization", true)

is_completed(cmdbuf::MTL.MTLCommandBufferLike) =
    cmdbuf.status >= MTL.MTLCommandBufferStatusCompleted

# blocking wait, performed on a worker thread by `cooperative_wait`. `@objc` calls are
# GC-safe, but the worker has no autorelease pool of its own, so set one up. it cannot be an
# `@autoreleasepool`, whose global lock may be held by the waiting task.
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

    if use_nonblocking_synchronization
        cooperative_wait(blocking_wait, cmdbuf; isdone=handlers ? nothing : is_completed)
    else
        wait_completed(cmdbuf)
    end
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
