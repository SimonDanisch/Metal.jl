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
