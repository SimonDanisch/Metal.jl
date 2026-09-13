# The control for research/mtl4_icb_stall.jl: same indirect command buffer, same
# kernel, same count, on the LEGACY queue. 60/60 submissions at 207 ms each.

# The control for the MTL4 stall: the SAME indirect command buffer and the same
# heavy kernel, replayed the same number of times, on the LEGACY queue.

using Metal
const MTL = Metal.MTL
say(a...) = (println(a...); flush(stdout))

dev = Metal.device()
const SPIN = parse(Int, get(ENV, "RAW_SPIN", "2048"))
const N    = parse(Int, get(ENV, "RAW_N", string(1 << 22)))
const ITER = parse(Int, get(ENV, "RAW_ITER", "60"))

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

arr = Metal.MtlVector{Float32}(undef, N); fill!(arr, 0f0); Metal.synchronize()
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

desc = MTL.MTLIndirectCommandBufferDescriptor(; max_kernel_buffers = 3, ray_tracing = true)
icb = MTL.MTLIndirectCommandBuffer(dev, desc, 1; storage = MTL.MTLResourceStorageModeShared)
c = MTL.indirect_compute_command(icb, 1)
MTL.set_pipeline!(c, kern.pipeline)
for (i, off) in enumerate((0, 256, 512)); MTL.set_kernel_buffer!(c, argbuf, off, i); end
MTL.dispatch_threadgroups!(c, Metal.MTLSize(cld(N, 256)), Metal.MTLSize(256))

bq = Metal.global_queue(dev)
Metal.make_persistently_resident!(argbuf)
Metal.make_persistently_resident!(icb, dev)

say("legacy  spin=", SPIN, " N=", N, " iters=", ITER)
t0 = time()
for i in 1:ITER
    enc = Metal.compute_encoder(bq)
    MTL.use!(enc, [argbuf], MTL.ReadWriteUsage)
    MTL.execute_commands!(enc, icb, 1:1)
    bq.last_pipeline = nothing
    Metal.note_recorded!(bq, 0, nothing)
    Metal.flush!(bq)
    Metal.synchronize()
    i % 10 == 0 && say("  i=", i, " t=", round(time() - t0, digits=2), "s")
end
say("LEGACY DONE  iters=", ITER, " secs=", round(time() - t0, digits=2),
    " per=", round((time() - t0) / ITER * 1e3, digits=1), "ms value=", Array(arr)[1])
