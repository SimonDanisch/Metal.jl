# A minimal reproducer for the MTL4 indirect-command-buffer stall.
#
# NOT run by the test suite: it hangs on purpose. Run it by hand.
#
#   RAW_SPIN=2048 julia --project research/mtl4_icb_stall.jl          # stalls at ~i=11
#   RAW_DIRECT=1  julia --project research/mtl4_icb_stall.jl          # same kernel, no ICB: fine
#   RAW_EMPTY=1   julia --project research/mtl4_icb_stall.jl          # no work at all: fine
#                 julia --project research/mtl4_icb_stall_control.jl  # legacy queue: fine
#   RAW_INDIRECTDISPATCH=1 julia --project research/mtl4_icb_stall.jl # the way out: fine
#
# `executeCommandsInBuffer:` on an MTL4ComputeCommandEncoder stops completing
# after roughly a second of cumulative replayed GPU work. The shared event stops
# advancing, `MTL4CommitFeedback` reports nothing, and the system log is empty.
# Heavier replays hit it sooner: at 207 ms per submission it stalls at the fifth,
# at 27 ms the twenty-first, and an empty submission never does.
#
# Every knob below was added to rule one suspected cause out, and NONE of them
# changes the outcome — all still stall at i=11 with sig one short:
#
#   RAW_FRESH=1          a new allocator AND command buffer every submission
#   RAW_NICB=40          forty distinct ICBs, so none is ever replayed twice
#   RAW_WAITEVERY=1      one submission in flight, host waits for each
#   RAW_FEEDBACK=0       no commit-feedback handler
#   RAW_RT=0             `ray_tracing` off on the ICB descriptor
#   RAW_PRIVATE=1        a Private ICB instead of Shared
#   RAW_BARRIER=1        an encoder `barrier!` after every execute
#   RAW_INDIRECTRANGE=1  the OTHER execute form, `indirectBuffer:`
#   RAW_RAWSEND=1        the selector sent by hand, bypassing the wrapper
#
# Three knobs DO change it, and only the last one is a way out:
#
#   RAW_DIRECT=1   the same kernel dispatched straight onto the MTL4 encoder
#                  through an argument table: 60/60 submissions, 208 ms each.
#                  So the stall is `executeCommandsInBuffer:` and not the queue,
#                  the kernel or the load. On its own it cannot express a
#                  `repeat!` gate, whose grid only the device knows.
#   RAW_FRESHQ=1   a new MTL4 queue per submission: no stall, and no work either
#                  (3 of 40 increments land). A second failure, not a fix.
#   RAW_INDIRECTDISPATCH=1
#                  direct encoding again, but with the threadgroup count READ
#                  FROM MEMORY: 60/60 submissions, 208 ms each. This is the one
#                  Mantle took — a closed gate writes a zero threadgroup count
#                  where a replay would write a zero-length execution range, so
#                  the gate survives and the host still never learns whether the
#                  iteration ran. Costs three sends per dispatch against one per
#                  segment. See `canreplay`/`encodes` in Mantle's
#                  `src/metal/device.jl`.

# Does a bare MTL4 submit loop stall, with no Mantle in it at all?
#
# One indirect command buffer, one heavy kernel, submitted in a loop: reset the
# allocator, refill the command buffer, execute, commit, signal, wait. That is
# the whole of what Mantle's MTL4 path does per frame, minus Mantle.

using Metal
const MTL = Metal.MTL
say(a...) = (println(a...); flush(stdout))

dev = Metal.device()
const SPIN = parse(Int, get(ENV, "RAW_SPIN", "8192"))
const N    = parse(Int, get(ENV, "RAW_N", string(1 << 22)))
const ITER = parse(Int, get(ENV, "RAW_ITER", "200"))
const RING = parse(Int, get(ENV, "RAW_RING", "3"))
const WAITEVERY = get(ENV, "RAW_WAITEVERY", "0") == "1"
const FRESH = get(ENV, "RAW_FRESH", "0") == "1"
const FRESHQ = get(ENV, "RAW_FRESHQ", "0") == "1"
const RAWSEND = get(ENV, "RAW_RAWSEND", "0") == "1"
const INDIRECTDISPATCH = get(ENV, "RAW_INDIRECTDISPATCH", "0") == "1"
const FEEDBACK = get(ENV, "RAW_FEEDBACK", "1") == "1"
const EMPTY = get(ENV, "RAW_EMPTY", "0") == "1"
const DIRECT = get(ENV, "RAW_DIRECT", "0") == "1"
const keepalive = Any[]

# The chain has to depend on MEMORY the compiler cannot prove constant, or it
# folds the whole loop away — the first version of this ran 120 submissions in
# 0.11s, which is not a heavy kernel, it is no kernel.
function heavy!(x, v::Float32)
    i = Metal.thread_position_in_grid_1d()
    n = length(x)
    acc = @inbounds x[i]
    for k in 1:SPIN
        j = (i + k) % n + 1
        acc = muladd(acc, 1.0000001f-3, @inbounds x[j])
    end
    @inbounds x[i] = acc * 0.0f0 + @inbounds(x[i]) + v
    return nothing
end

arr = Metal.MtlVector{Float32}(undef, N)
fill!(arr, 0f0); Metal.synchronize()

kern = Metal.mtlfunction(heavy!, Tuple{Metal.MtlDeviceVector{Float32,1}, Float32};
                         name = "heavy", indirect = true)
argbuf = MTL.MTLBuffer(dev, 1024; storage = Metal.SharedStorage)
base = convert(Ptr{UInt8}, MTL.contents(argbuf))
_, ma = Metal.malloc_buffer_and_gpu_address(dev)
_, ea = Metal.exception_info_buffer_and_gpu_address(dev)
st = Metal.KernelState(UInt32(1),
    reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, ma),
    reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, ea),
    reinterpret(Core.LLVMPtr{UInt64, Metal.AS.Device}, UInt64(0)))
put(x, off) = (r = Ref(x); GC.@preserve r unsafe_copyto!(base + off,
    convert(Ptr{UInt8}, Base.unsafe_convert(Ptr{typeof(x)}, r)), sizeof(typeof(x))))
put(st, 0); put(Metal.mtlconvert(arr), 256); put(1.0f0, 512)

const NICB    = parse(Int, get(ENV, "RAW_NICB", "1"))
const RT      = get(ENV, "RAW_RT", "1") == "1"
const PRIVATE = get(ENV, "RAW_PRIVATE", "0") == "1"
const BARRIER = get(ENV, "RAW_BARRIER", "0") == "1"
const OPT     = get(ENV, "RAW_OPT", "0") == "1"
const INDIRECTRANGE = get(ENV, "RAW_INDIRECTRANGE", "0") == "1"
desc = MTL.MTLIndirectCommandBufferDescriptor(; max_kernel_buffers = 3, ray_tracing = RT)
icbs = map(1:NICB) do _
    b = MTL.MTLIndirectCommandBuffer(dev, desc, 1;
        storage = PRIVATE ? MTL.MTLResourceStorageModePrivate : MTL.MTLResourceStorageModeShared)
    c = MTL.indirect_compute_command(b, 1)
    MTL.set_pipeline!(c, kern.pipeline)
    for (i, off) in enumerate((0, 256, 512)); MTL.set_kernel_buffer!(c, argbuf, off, i); end
    MTL.dispatch_threadgroups!(c, Metal.MTLSize(cld(N, 256)), Metal.MTLSize(256))
    b
end
icb = icbs[1]

lq = Metal.global_queue(dev).queue
rs = Metal.install_queue_residency!(lq, dev)
for r in (argbuf, arr.data[]); MTL.add_allocation!(rs, r); end
for b in icbs; MTL.add_allocation!(rs, b); end
MTL.commit!(rs)

# An argument table holding the same three slots the ICB commands bind.
table = MTL.MTL4ArgumentTable(dev; buffers = 4)
let gpu = UInt64(argbuf.gpuAddress)
    for (i, off) in enumerate((0, 256, 512))
        MTL.set_address!(table, gpu + UInt64(off), i)
    end
end

# A threadgroup count the device could have written: the indirect-dispatch form
# of a gate. Three UInt32, and zero in the first closes the gate.
gridbuf = MTL.MTLBuffer(dev, 32; storage = Metal.SharedStorage)
let p32 = convert(Ptr{UInt32}, MTL.contents(gridbuf))
    unsafe_store!(p32, UInt32(cld(N, 256)), 1)
    unsafe_store!(p32, UInt32(1), 2); unsafe_store!(p32, UInt32(1), 3)
end

# An execution range the device could have written: location 0, length 1.
rangebuf = MTL.MTLBuffer(dev, 16; storage = Metal.SharedStorage)
let p32 = convert(Ptr{UInt32}, MTL.contents(rangebuf))
    unsafe_store!(p32, UInt32(0), 1); unsafe_store!(p32, UInt32(1), 2)
end
MTL.add_allocation!(rs, rangebuf); MTL.add_allocation!(rs, gridbuf); MTL.commit!(rs)

q  = MTL.MTL4CommandQueue(dev)
MTL.add_residency_set!(q, rs)
allocs = [MTL.MTL4CommandAllocator(dev) for _ in 1:RING]
cbs    = [MTL.MTL4CommandBuffer(dev) for _ in 1:RING]
at     = zeros(UInt64, RING)
ev = MTL.MTLSharedEvent(dev)
nxt = UInt64(0)
fb = MTL.MTL4Feedback()

say("spin=", SPIN, " N=", N, " ring=", RING, " waitevery=", WAITEVERY,
    " rt=", RT, " private=", PRIVATE, " barrier=", BARRIER, " nicb=", NICB,
    " direct=", DIRECT, " empty=", EMPTY)
t0 = time()
for i in 1:ITER
    slot = mod1(i, RING)
    f = at[slot]
    if !iszero(f) && ev.signaledValue < f
        tw = time()
        while ev.signaledValue < f && time() - tw < 20; yield(); end
        if ev.signaledValue < f
            say("STALL at i=", i, " waiting slot=", slot, " for ", Int(f),
                " sig=", Int(ev.signaledValue), " next=", Int(nxt),
                " after ", round(time() - t0, digits=2), "s")
            say("  feedback: ", something(MTL.failure(fb), "none"))
            exit(1)
        end
    end
    # RAW_FRESH: never reuse an allocator or a command buffer, so `reset!` timing
    # cannot be the cause. Leaks by design — this is a diagnostic.
    if FRESH
        al = MTL.MTL4CommandAllocator(dev); cb = MTL.MTL4CommandBuffer(dev)
        push!(keepalive, al); push!(keepalive, cb)
    else
        MTL.reset!(allocs[slot])
        al = allocs[slot]; cb = cbs[slot]
    end
    MTL.begin_command_buffer!(cb, al)
    enc = MTL.compute_encoder(cb)
    MTL.use_residency_set!(cb, rs)
    if EMPTY
        # nothing
    elseif INDIRECTDISPATCH
        # Direct encoding, but the GRID comes from memory: the gate mechanism a
        # recorded plan needs without an indirect command buffer. A closed gate
        # writes zero threadgroups instead of a zero-length execution range.
        MTL.set_argument_table!(enc, table)
        MTL.set_function!(enc, kern.pipeline)
        MTL.dispatch_threadgroups_indirect!(enc, gridbuf, 0, Metal.MTLSize(256))
        BARRIER && MTL.barrier!(enc)
    elseif DIRECT
        # The same kernel, dispatched straight onto the MTL4 encoder through an
        # argument table, so the only difference from the branch below is that no
        # indirect command buffer is involved.
        MTL.set_argument_table!(enc, table)
        MTL.set_function!(enc, kern.pipeline)
        MTL.dispatch_threadgroups!(enc, Metal.MTLSize(cld(N, 256)), Metal.MTLSize(256))
    elseif INDIRECTRANGE
        # The OTHER execute form: the command processor reads (location, length)
        # from memory instead of taking it from the encoder. Different path in
        # the driver, and the one a gated plan already uses.
        MTL.execute_commands_indirect!(enc, icbs[mod1(i, NICB)], rangebuf, 0)
        BARRIER && MTL.barrier!(enc)
    elseif RAWSEND
        # Bypass the wrapper entirely: the selector sent by hand, so the stall
        # cannot be blamed on how Metal.jl marshals an NSRange.
        b = icbs[mod1(i, NICB)]
        ccall(:objc_msgSend, Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, MTL.NSRange),
              reinterpret(Ptr{Cvoid}, pointer(enc)),
              ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                    "executeCommandsInBuffer:withRange:"),
              reinterpret(Ptr{Cvoid}, pointer(b)), MTL.NSRange(0, 1))
        BARRIER && MTL.barrier!(enc)
    else
        MTL.execute_commands!(enc, icbs[mod1(i, NICB)], 1:1)
        BARRIER && MTL.barrier!(enc)
    end
    MTL.endEncoding!(enc)
    MTL.end_command_buffer!(cb)
    # RAW_FRESHQ: a brand new MTL4 queue per submission, to ask whether the limit
    # is per QUEUE. If it is, rotating queues is a workaround that keeps the
    # indirect command buffer — and so `repeat!` gates — working.
    if FRESHQ
        global q = MTL.MTL4CommandQueue(dev)
        MTL.add_residency_set!(q, rs)
        push!(keepalive, q)
    end
    FEEDBACK ? MTL.commit!(q, [cb], fb) : MTL.commit!(q, [cb])
    global nxt += UInt64(1)
    MTL.signal_event!(q, ev, nxt)
    at[slot] = nxt
    if WAITEVERY
        tw = time()
        while ev.signaledValue < nxt && time() - tw < 20; yield(); end
        ev.signaledValue < nxt && (say("STALL waiting own ", Int(nxt),
            " sig=", Int(ev.signaledValue), " after ", round(time()-t0, digits=2), "s");
            say("  feedback: ", something(MTL.failure(fb), "none")); exit(1))
    end
    i % 20 == 0 && say("  i=", i, " sig=", Int(ev.signaledValue),
                       " t=", round(time() - t0, digits=2), "s")
end
tw = time(); while ev.signaledValue < nxt && time() - tw < 30; yield(); end
say(ev.signaledValue >= nxt ? "NO STALL" : "STALL at drain",
    "  submissions=", Int(nxt), " sig=", Int(ev.signaledValue),
    " secs=", round(time() - t0, digits=2), " value=", Array(arr)[1])
