export device, device!, global_queue, BatchedCommandQueue

log_compiler()          = OSLog("org.juliagpu.metal", "Compiler")
log_compiler(args...)   = log_compiler()(args...)
log_array()             = OSLog("org.juliagpu.metal", "Array")
log_array(args...)      = log_array()(args...)

const LABEL_RESOURCES = @load_preference("label_resources", nothing)

@inline label_resources() = @something(LABEL_RESOURCES, Base.JLOptions().debug_level >= 2)

macro label!(obj, str)
    quote
        if label_resources()
            $(esc(obj)).label = $(esc(str))
        end
        nothing
    end
end

"""
    device()::MTLDevice

Return the Metal GPU device associated with the current Julia task.

Since all M-series systems currently only externally show a single GPU, this function
effectively returns the only system GPU.
"""
function device()
    get!(task_local_storage(), :MTLDevice) do
        dev = MTLDevice(1)
        if is_virtual(dev) && macos_version() >= v"15"
            @warn """Metal.jl is running on a virtualized Apple GPU; this is supported on a
                     best-effort basis, so you may run into issues.""" maxlog=1
        elseif is_virtual(dev) && macos_version() < v"15"
            @error "Metal.jl does not support virtualized Apple GPUs below macOS 15." maxlog=1
        elseif !supports_family(dev, MTL.MTLGPUFamilyApple7) ||
               !supports_family(dev, MTL.MTLGPUFamilyMetal3)
            @error "Metal.jl is only supported on Metal 3-capable Apple Silicon (M-series) GPUs." maxlog=1
        end
        return dev
    end::MTLDevice
end

"""
    device!(dev::MTLDevice)

Sets the Metal GPU device associated with the current Julia task.
"""
device!(dev::MTLDevice) = task_local_storage(:MTLDevice, dev)

const global_queues = WeakKeyDict{Any,Nothing}()
const global_queues_lock = ReentrantLock()

function active_global_queues()
    Base.@lock global_queues_lock collect(keys(global_queues))
end

"""
    global_queue(dev::MTLDevice)::BatchedCommandQueue

Return the [`BatchedCommandQueue`](@ref) associated with the current Julia task.

This is a *batched* queue: kernel launches and blit operations accumulate into a
single command buffer and are submitted lazily, rather than one command buffer per
operation. It is a drop-in for a raw `MTLCommandQueue` — using it as one (e.g. to
derive a command buffer, or for MPS) preserves program order by draining pending
batches when command buffers are enqueued or committed. Call [`synchronize`](@ref)
to wait for submitted work to finish. See
[`BatchedCommandQueue`](@ref) for the full draining semantics.

Two lookups and no allocation on the hit path. It used to be one
`get!(task_local_storage(), (:BatchedCommandQueue, dev)) do … end`, which allocates
TWICE per call before it can even look: the `(Symbol, MTLDevice)` key is a tuple that
has to be boxed to enter the store, and the `do` block is a closure over `dev`. Every
kernel launch asks this, so a Hikari sample paid for it hundreds of times — 576 of the
8256 bytes a 400-launch allocation profile attributed to Metal.jl.

The nesting is what removes it: a `Symbol` key is a singleton, so the outer lookup
boxes nothing, and the inner table is keyed by the device's POINTER.

A pointer and not the device itself, because `MTLDevice` is an isbits immutable — an
`IdDict{MTLDevice,…}` takes its key as `Any` and so boxes one on every lookup, 16 bytes
a launch. A `Dict{UInt,…}` hashes the pointer with nothing to box. The device is a
system singleton that outlives every queue made from it, so its address is a stable
name for it.
"""
const adopted_key = Ref(UInt(0))
const adopted_queue = Ref{Any}(nothing)

function global_queue(dev::MTLDevice)
    key = UInt(pointer(dev))
    # An adopted queue is the device's, not the task's, and answering it first is
    # the whole point: a caller that owns submission has ONE queue and therefore
    # one residency set, and a task that happened not to have made a queue yet
    # must not get a second. One integer compare on the launch path, and `0` is
    # not a device address, so an un-adopted process never takes the branch.
    adopted_key[] == key && return adopted_queue[]::BatchedCommandQueue
    queues = task_queues()
    bq = get(queues, key, nothing)
    bq === nothing || return bq::BatchedCommandQueue
    return make_task_queue!(queues, key, dev)
end

"""
    adopt_queue!(dev, bq) -> bq

Make `bq` the queue every task uses for `dev`, instead of one of its own.

`global_queue` hands every task a private `BatchedCommandQueue` so that
independent tasks never contend for one command buffer. That default is wrong for
a caller that owns submission itself: a render graph has ONE queue per device, its
residency set belongs to that queue, and work that lands on a second queue is
ordered against the first by nothing at all. Silently, at that — a buffer made
resident on one queue's set and read from another's is not an error; the reads
come back zero and the writes are dropped.

PROCESS-WIDE and not per task, because the uploads and the replay have to meet:
a `Buffer(dev, data)` built on one task and a plan run on another are the same
graph, and adopting per task would put the blit on a queue the replay is ordered
against by nothing.

The caller taking this over is also taking over the rule the default enforced: a
`BatchedCommandQueue` is mutated without a lock, so an adopted queue must be
driven by one task at a time.
"""
function adopt_queue!(dev::MTLDevice, bq)
    key = UInt(pointer(dev))
    # The steady state: `openrun` asks every frame and the answer has not changed.
    adopted_key[] == key && adopted_queue[] === bq && return bq
    # A CHANGE, though, has to drain. Metal orders command buffers only within one
    # queue, so everything already committed to the queue being left is ordered
    # against everything the new one will run by nothing at all. The case is not
    # hypothetical: a `Buffer(dev, data)` built before its device adopted the queue
    # is a blit on whatever queue the task had, and it landed AFTER the first frame
    # that read the buffer — one wrong frame, every frame after it right, and
    # nothing reported. Draining here is the only moment both queues are known.
    device_synchronize()
    adopted_queue[] = bq
    adopted_key[] = key
    return bq
end

"""This task's device-to-queue table, made on first use."""
@inline function task_queues()
    tls = task_local_storage()
    q = get(tls, :metal_task_queues, nothing)
    q === nothing || return q::Dict{UInt,Any}
    fresh = Dict{UInt,Any}()
    tls[:metal_task_queues] = fresh
    return fresh
end

# `Dict{UInt,Any}` and not a value type of `BatchedCommandQueue`: a SIGNATURE is
# evaluated when the method is defined, and `state.jl` is included before
# `command_batching.jl`, so naming that type here fails to load the package. Inside a
# body the same name resolves at call time, which is why the assertions above spell it.
@noinline function make_task_queue!(queues::Dict{UInt,Any}, key::UInt, dev::MTLDevice)
    return get!(queues, key) do
        @autoreleasepool begin
            # NOTE: MTLCommandQueue itself is manually reference-counted,
            #       the release pool is for resources used during its construction.
            queue = MTLCommandQueue(dev)
            queue.label = "global_queue($(current_task()))"
            bq = BatchedCommandQueue(queue)
            task_local_storage(batched_queue_key(queue), bq)
            Base.@lock global_queues_lock global_queues[bq] = nothing
            bq
        end
    end::BatchedCommandQueue
end

# tracks the most recently launched logging-enabled cmdbuf per queue, so that
# `synchronize` can wait on it and thereby drain its `addLogHandler:` blocks
# (Metal dispatches log delivery asynchronously and offers no flush primitive;
# `waitUntilCompleted` on the specific cmdbuf is what processes its pending blocks).
const logging_cmdbufs = IdDict{MTLCommandQueue,MTLCommandBuffer}()
const logging_cmdbufs_lock = ReentrantLock()

function track_logging_cmdbuf!(queue::MTLCommandQueue, cmdbuf::MTLCommandBuffer)
    Base.@lock logging_cmdbufs_lock begin
        logging_cmdbufs[queue] = cmdbuf
    end
    return
end

function drain_logging_cmdbufs!(queue::MTLCommandQueue)
    cmdbuf = Base.@lock logging_cmdbufs_lock begin
        prev = get(logging_cmdbufs, queue, nothing)
        delete!(logging_cmdbufs, queue)
        prev
    end
    if cmdbuf !== nothing
        MTL.wait_completed(cmdbuf)
    end
    return
end


## scratch-buffer residency

# Fast residency path; collapse this to `true` when macOS 14 support is dropped.
function can_use_residency_sets(dev::MTLDevice)
    @memoize key=pointer(dev)::id{MTLDevice} begin
        is_macos(v"15") && !is_virtual(dev)
    end::Bool
end

const queue_residency_sets = Dict{UInt,MTLResidencySet}()
const queue_residency_sets_lock = ReentrantLock()

"""
What each residency set already holds, as `set pointer => allocation pointers`.

`addAllocation:` is idempotent to Metal but not free: the call and the `commit`
that has to follow it are driver work, and `make_persistently_resident!` is
reached from `adapt_storage` — once per baked address per LAUNCH, not once per
buffer as its docstring assumed. Measured on a still Hikari frame: 1382 calls per
sample, the largest single source of allocation in the render.

A pointer is a safe key only because an allocation leaves a set exactly once, when
it is FREED (`forget_resident!`), and its memory is released after that — so an
address inside a set cannot be reused by a later buffer while the set still names
it.

This comment used to say "nothing ever removes an allocation from these sets", and
that was the leak: a set holds a STRONG reference to everything in it, so a buffer
made resident once was immortal however many times Julia freed it. Qwen-Image
2.1's denoiser is 7.26 GB of weights, every one of them reached by a baked address
and therefore resident; releasing the model gave back nothing. Measured on this
machine: 1170 allocations totalling 22.0 GB, 775 of them freed totalling 16.1 GB,
and the device still reporting 15.1 GiB in use.
"""
const residency_members = Dict{UInt,Set{UInt}}()

"""
Buffers freed while `queue_residency_sets_lock` was busy, to be taken out of their
set by the next caller that gets it.

`free` runs from a FINALIZER, and a finalizer that blocks on a lock another thread
is inside deadlocks. So the free path only ever `trylock`s, and what it cannot do
now it leaves here. Deferring is safe: the set's own reference is what keeps the
allocation alive, so a buffer waiting here is not dangling — it is merely still
resident, which is the state it was already in.
"""
const pending_residency_drops = Vector{Any}()
const pending_residency_lock = ReentrantLock()

"""Buffers neither dropped nor deferred, because both `trylock`s lost. Observable
rather than silent: a number that grows is this scheme failing."""
const residency_drop_misses = Threads.Atomic{Int}(0)

"""
    forget_resident!(buf)

Take `buf` out of every residency set that holds it, so that releasing it actually
frees it. Called by [`free`](@ref), which runs from a finalizer — hence `trylock`
throughout and [`pending_residency_drops`](@ref) for what has to wait.
"""
function forget_resident!(buf)
    if trylock(queue_residency_sets_lock)
        try
            drop_resident_locked!(buf)
            flush_pending_drops_locked!()
        finally
            unlock(queue_residency_sets_lock)
        end
        return nothing
    end
    if trylock(pending_residency_lock)
        try
            push!(pending_residency_drops, buf)
        finally
            unlock(pending_residency_lock)
        end
        return nothing
    end
    Threads.atomic_add!(residency_drop_misses, 1)
    return nothing
end

"""One buffer out of whichever set holds it. `queue_residency_sets_lock` held."""
function drop_resident_locked!(buf)
    p = UInt(pointer(buf))
    for (_, resset) in queue_residency_sets
        members = get(residency_members, UInt(pointer(resset)), nothing)
        members === nothing && continue
        p in members || continue
        delete!(members, p)
        MTL.remove_allocation!(resset, buf)
        # One `commit` per set per call. Batched with the deferred ones below,
        # which is why they are flushed from here rather than each on its own.
        MTL.commit!(resset)
    end
    return nothing
end

"""Everything that had to wait. `queue_residency_sets_lock` held."""
function flush_pending_drops_locked!()
    isempty(pending_residency_drops) && return 0
    trylock(pending_residency_lock) || return 0
    drops = try
        d = copy(pending_residency_drops); empty!(pending_residency_drops); d
    finally
        unlock(pending_residency_lock)
    end
    for b in drops
        drop_resident_locked!(b)
    end
    return length(drops)
end

"""
    trim_residency!() -> Int

Take every freed-but-still-resident buffer out of its set, now, and say how many.

The free path defers whatever it could not do without blocking a finalizer; this
is the caller saying "I have finished with a model and want the memory back" —
the same explicit trim a pool has, for the same reason.
"""
function trim_residency!()
    Base.@lock queue_residency_sets_lock flush_pending_drops_locked!()
end

command_queue_key(queue::MTLCommandQueue) = UInt(pointer(queue))

function install_queue_residency!(queue::MTLCommandQueue, dev::MTLDevice)
    key = command_queue_key(queue)
    Base.@lock queue_residency_sets_lock begin
        cached_resset = get(queue_residency_sets, key, nothing)
        cached_resset === nothing || return cached_resset
    end

    malloc_buf = malloc_buffer(dev)
    exc_buf = exception_info_buffer(dev)

    Base.@lock queue_residency_sets_lock begin
        cached_resset = get(queue_residency_sets, key, nothing)
        cached_resset === nothing || return cached_resset

        desc = MTLResidencySetDescriptor()
        desc.initialCapacity = 2
        @label! desc "Metal scratch buffers"

        resset = MTLResidencySet(dev, desc)
        MTL.add_allocation!(resset, malloc_buf)
        MTL.add_allocation!(resset, exc_buf)
        MTL.commit!(resset)
        MTL.add_residency_set!(queue, resset)
        queue_residency_sets[key] = resset
        finalizer(queue) do _
            Base.@lock queue_residency_sets_lock begin
                get(queue_residency_sets, key, nothing) === resset &&
                    delete!(queue_residency_sets, key)
            end
        end
        return resset
    end
end


## dynamic-memory allocator buffer
#
# kernels that perform dynamic memory allocations bump-allocate out of a per-
# device scratch buffer. the buffer is allocated lazily on first use, and its
# counter is never reset; allocations are monotonic for the device's lifetime
# (the bump-allocator-style design is intentional, see device/malloc.jl).
#
# 1 MB allows ~65k 16-byte boxes before exhaustion, plenty for the dead
# throw-path boxing that motivates this.

const MALLOC_BUF_SIZE = 1024 * 1024

const device_malloc_bufs = Dict{MTLDevice, Tuple{MTLBuffer, UInt64}}()
const device_malloc_lock = ReentrantLock()

function malloc_buffer_and_gpu_address(dev::MTLDevice)
    Base.@lock device_malloc_lock begin
        get!(device_malloc_bufs, dev) do
            buf = @autoreleasepool MTLBuffer(dev, MALLOC_BUF_SIZE;
                                             storage=SharedStorage)
            # initialize the counter (first 4 bytes) to 4 so the first
            # allocation lands past the counter itself
            unsafe_store!(convert(Ptr{UInt32}, MTL.contents(buf)), UInt32(4))
            (buf, UInt64(buf.gpuAddress))
        end
    end
end

malloc_buffer(dev::MTLDevice) = first(malloc_buffer_and_gpu_address(dev))
