# Retiring command buffers costs the same however many finished.
#
# `PendingCommand.cmdbuf` is the abstract `MTLCommandBufferLike`, and
# `drain_cleanups!` read `.status` off it inline: a dynamic call that boxed the
# returned enum, 40 B for every command buffer it retired. A frame's allocations
# then depended on how many buffers happened to finish during it, which is what
# made RayMakie's zero-allocation sample test fail one sample in twenty.

@testset "drain_cleanups! does not allocate per retired command buffer" begin
    bq = Metal.BatchedCommandQueue(MTLCommandQueue(device()))
    function drained_bytes(bq, n)
        for _ in 1:n
            cb = MTLCommandBuffer(bq.queue)
            commit!(cb)
            wait_completed(cb)
            Metal.defer_cleanup!(bq, cb, Any[])
        end
        @test Metal.pending_cleanup_count(bq) == n
        bytes = @allocated Metal.drain_cleanups!(bq)
        @test Metal.pending_cleanup_count(bq) == 0
        return bytes
    end
    # More than the in-flight cap first, so the recycled-roots list is at its
    # working size and its growth is not what gets measured.
    drained_bytes(bq, 4 * bq.inflight)
    @test drained_bytes(bq, 16) == drained_bytes(bq, 1)
end

# A command buffer committed on an ADOPTED queue runs after the work still open in
# its batch, as it does on a task's own queue. The commit hook looked the batch up
# in task-local storage, where an adopted batch never is (it is every task's), so
# it flushed nothing: here the blit's 3 ran first and the second launch's 2 landed
# on top of it. `batched_queue` of the raw queue made a second wrapper around it
# for the same reason.
@testset "an adopted queue orders a raw command buffer after its open batch" begin
    function write_kernel(A, value)
        A[1] = value
        return
    end
    dev = device()
    bq = Metal.BatchedCommandQueue(MTLCommandQueue(dev))
    prev_key, prev_queue = Metal.adopted_key[], Metal.adopted_queue[]
    try
        Metal.adopt_queue!(dev, bq)
        @test Metal.global_queue(dev) === bq
        @test Metal.batched_queue(bq.queue) === bq

        D = MtlArray(UInt8[0])
        @metal threads=1 queue=bq write_kernel(D, UInt8(1))
        cmdbuf = MTLCommandBuffer(bq.queue)
        @metal threads=1 queue=bq write_kernel(D, UInt8(2))
        MTL.MTLBlitCommandEncoder(cmdbuf) do enc
            buf = Base.unsafe_convert(MTL.MTLBuffer, D)
            MTL.append_fillbuffer!(enc, buf, UInt8(3), sizeof(D), D.offset)
        end
        commit!(cmdbuf)
        synchronize(bq)
        @test Array(D) == UInt8[3]
    finally
        Metal.device_synchronize()
        Metal.adopted_queue[] = prev_queue
        Metal.adopted_key[] = prev_key
    end
end
