# Recording a plan once and replaying it, which on Metal means an INDIRECT command
# buffer.
#
# An `MTLCommandBuffer` is single-use: it cannot be committed twice, so the way a
# Vulkan backend bakes a render graph — record a `VkCommandBuffer` once, re-submit it
# every frame — has no counterpart built out of one. `MTLIndirectCommandBuffer` is the
# counterpart: commands are encoded into it once and a normal encoder replays the
# range with `executeCommandsInBuffer:withRange:` for as long as the plan lives.
#
# Three things have to hold for a graph to be baked onto it, and each is a test here:
#
#   1. a kernel THIS package compiled can be encoded into a command and replayed;
#   2. its arguments can come from a buffer, because a command may only
#      `setKernelBuffer` — there is no `setBytes` on an indirect command, so the
#      arguments have to be packed once into memory the command points at;
#   3. commands are CONCURRENT unless told otherwise, so a pass boundary needs
#      `setBarrier`.

using Test, Metal
const MTLi = Metal.MTL

function icb_add_const!(a, v::Float32)
    i = Metal.thread_position_in_grid_1d()
    @inbounds a[i] += v
    return nothing
end

"""A pipeline that may be used from an indirect command buffer.

`supportIndirectCommandBuffers` is not a hint: a pipeline without it is refused by
`setComputePipelineState` on a command, which is why `@metal`'s pipeline cannot be
used here — it does not set it. `mtlfunction`'s `indirect` keyword builds the same
kernel with the flag, so the compiled code, the relocation table and the linked
functions are the ordinary path's.
"""
function icb_pipeline(f, tt, name)
    kernel = Metal.mtlfunction(f, tt; name, indirect = true)
    return kernel.pipeline, kernel
end

"""
The argument block a command binds, packed once.

The layout is what `launch` pushes per call with `set_bytes!`: the kernel state at
index 1, then one slot per argument. A buffer bound at index `i` and bytes pushed at
index `i` are the same binding to the shader, which is what lets a recorded command
carry arguments at all.
"""
function icb_pack(dev, out, value::Float32)
    argbuf = MTLi.MTLBuffer(dev, 1024; storage = Metal.SharedStorage)
    base = convert(Ptr{UInt8}, MTLi.contents(argbuf))
    _, maddr = Metal.malloc_buffer_and_gpu_address(dev)
    _, eaddr = Metal.exception_info_buffer_and_gpu_address(dev)
    state = Metal.KernelState(UInt32(1),
        reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, maddr),
        reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, eaddr),
        reinterpret(Core.LLVMPtr{UInt64, Metal.AS.Device}, UInt64(0)))
    function put(x, off)
        r = Ref(x)
        GC.@preserve r unsafe_copyto!(base + off,
            convert(Ptr{UInt8}, Base.unsafe_convert(Ptr{typeof(x)}, r)), sizeof(x))
    end
    # 256-byte slots: a buffer offset has an alignment rule, and one slot per argument
    # keeps the packing readable. A real plan packs tightly against `argsize`.
    put(state, 0)
    put(Metal.mtlconvert(out), 256)
    put(value, 512)
    return argbuf
end

"""Encode `n` commands, all the same dispatch, optionally serialised."""
function icb_encode(dev, pip, argbuf, n; barrier::Bool)
    desc = MTLi.MTLIndirectCommandBufferDescriptor(; max_kernel_buffers = 4)
    icb = MTLi.MTLIndirectCommandBuffer(dev, desc, n)
    for i in 1:n
        c = MTLi.indirect_compute_command(icb, i)
        MTLi.set_pipeline!(c, pip)
        MTLi.set_kernel_buffer!(c, argbuf, 0, 1)
        MTLi.set_kernel_buffer!(c, argbuf, 256, 2)
        MTLi.set_kernel_buffer!(c, argbuf, 512, 3)
        MTLi.dispatch_threadgroups!(c, Metal.MTLSize(1), Metal.MTLSize(64))
        barrier && MTLi.set_barrier!(c)
    end
    return icb
end

"""One replay: the whole of a baked frame's host work."""
function icb_replay!(queue, icb, n, resources)
    cb = MTLi.MTLCommandBuffer(queue)
    enc = MTLi.MTLComputeCommandEncoder(cb)
    # The commands reach their buffers by ADDRESS, and an address the encoder was
    # never told about is not resident — it reads as zeros rather than faulting.
    for r in resources
        MTLi.use!(enc, r, MTLi.ReadWriteUsage)
    end
    MTLi.execute_commands!(enc, icb, 1:n)
    MTLi.endEncoding!(enc)
    MTLi.commit!(cb)
    MTLi.wait_completed(cb)
    return nothing
end

@testset "a compiled kernel replays out of an indirect command buffer" begin
    dev = Metal.device()
    pip, lib = icb_pipeline(icb_add_const!,
                            Tuple{Metal.MtlDeviceVector{Float32,1}, Float32}, "icb_add")
    out = Metal.MtlVector{Float32}(undef, 64)
    fill!(out, 0f0); Metal.synchronize()
    argbuf = icb_pack(dev, out, 2.5f0)
    icb = icb_encode(dev, pip, argbuf, 1; barrier = false)
    queue = MTLi.MTLCommandQueue(dev)
    mbuf, _ = Metal.malloc_buffer_and_gpu_address(dev)
    ebuf, _ = Metal.exception_info_buffer_and_gpu_address(dev)
    res = (argbuf, mbuf, ebuf, out.data[])

    # Encoded ONCE, above. Each replay runs it again with nothing re-encoded, which
    # is the property the whole recording path rests on.
    icb_replay!(queue, icb, 1, res)
    @test all(==(2.5f0), Array(out))
    icb_replay!(queue, icb, 1, res)
    @test all(==(5.0f0), Array(out))
    icb_replay!(queue, icb, 1, res)
    @test all(==(7.5f0), Array(out))

    GC.@preserve lib nothing
end

@testset "commands in one buffer are concurrent until a barrier says otherwise" begin
    dev = Metal.device()
    pip, lib = icb_pipeline(icb_add_const!,
                            Tuple{Metal.MtlDeviceVector{Float32,1}, Float32}, "icb_add2")
    queue = MTLi.MTLCommandQueue(dev)
    mbuf, _ = Metal.malloc_buffer_and_gpu_address(dev)
    ebuf, _ = Metal.exception_info_buffer_and_gpu_address(dev)
    n = 32

    # WITH barriers: 32 read-modify-writes of the same memory, one after another, so
    # the count is exact. This is what a pass boundary has to encode.
    out = Metal.MtlVector{Float32}(undef, 64); fill!(out, 0f0); Metal.synchronize()
    argbuf = icb_pack(dev, out, 1f0)
    icb = icb_encode(dev, pip, argbuf, n; barrier = true)
    icb_replay!(queue, icb, n, (argbuf, mbuf, ebuf, out.data[]))
    @test all(==(Float32(n)), Array(out))

    # WITHOUT: the same 32 commands race, and the sum comes out short. Asserted rather
    # than left implicit, because it is the reason `set_barrier!` exists — a plan whose
    # passes are recorded without one would lose writes exactly like this, and silently.
    out2 = Metal.MtlVector{Float32}(undef, 64); fill!(out2, 0f0); Metal.synchronize()
    argbuf2 = icb_pack(dev, out2, 1f0)
    icb2 = icb_encode(dev, pip, argbuf2, n; barrier = false)
    icb_replay!(queue, icb2, n, (argbuf2, mbuf, ebuf, out2.data[]))
    @test maximum(Array(out2)) < Float32(n)

    GC.@preserve lib nothing
end

# The range writer: one thread, deciding at EXECUTION time how much of the buffer
# the next replay runs. `flag` is what a `repeat!` gate writes; `len` is how many
# commands the iteration holds.
function icb_write_range!(range, flag, loc::UInt32, len::UInt32)
    @inbounds range[1] = loc
    @inbounds range[2] = flag[1] != 0 ? len : UInt32(0)
    return nothing
end

@testset "two replays on one encoder run in order" begin
    dev = Metal.device()
    pip, lib = icb_pipeline(icb_add_const!,
                            Tuple{Metal.MtlDeviceVector{Float32,1}, Float32}, "icb_add3")
    queue = MTLi.MTLCommandQueue(dev)
    mbuf, _ = Metal.malloc_buffer_and_gpu_address(dev)
    ebuf, _ = Metal.exception_info_buffer_and_gpu_address(dev)
    n = 32

    # The same 32 read-modify-writes as the concurrency test, but as 32 separate
    # `executeCommandsInBuffer` calls with NO barrier encoded in any command. A
    # compute encoder is serial unless it was made concurrent, and a replay is one
    # command in it — so the sum is exact where a single replay of the same 32
    # raced and came out short. This is what a recorded plan's pass boundaries
    # rest on: separate replays need no barrier of their own.
    out = Metal.MtlVector{Float32}(undef, 64); fill!(out, 0f0); Metal.synchronize()
    argbuf = icb_pack(dev, out, 1f0)
    icb = icb_encode(dev, pip, argbuf, n; barrier = false)

    cb = MTLi.MTLCommandBuffer(queue)
    enc = MTLi.MTLComputeCommandEncoder(cb)
    for r in (argbuf, mbuf, ebuf, out.data[])
        MTLi.use!(enc, r, MTLi.ReadWriteUsage)
    end
    for i in 1:n
        MTLi.execute_commands!(enc, icb, i:i)
    end
    MTLi.endEncoding!(enc)
    MTLi.commit!(cb)
    MTLi.wait_completed(cb)
    @test all(==(Float32(n)), Array(out))

    GC.@preserve lib nothing
end

@testset "a device-written execution range decides what a replay runs" begin
    dev = Metal.device()
    queue = MTLi.MTLCommandQueue(dev)
    mbuf, _ = Metal.malloc_buffer_and_gpu_address(dev)
    ebuf, _ = Metal.exception_info_buffer_and_gpu_address(dev)
    addpip, lib1 = icb_pipeline(icb_add_const!,
                                Tuple{Metal.MtlDeviceVector{Float32,1}, Float32}, "icb_add4")
    rngpip, lib2 = icb_pipeline(icb_write_range!,
                                Tuple{Metal.MtlDeviceVector{UInt32,1},
                                      Metal.MtlDeviceVector{UInt32,1}, UInt32, UInt32},
                                "icb_range")

    out = Metal.MtlVector{Float32}(undef, 64)
    # Two elements each, not one: a four-byte buffer an indirect command reaches by
    # ADDRESS is not made resident by `useResource` — see the last testset here.
    range = Metal.MtlVector{UInt32}(undef, 2)
    flag = Metal.MtlVector{UInt32}(undef, 2)
    ngated = 4

    # One buffer, two kinds of command: slot 1 writes the execution range, slots
    # 2:5 are the gated work. Encoded ONCE for both halves of the test.
    argbuf = MTLi.MTLBuffer(dev, 2048; storage = Metal.SharedStorage)
    base = convert(Ptr{UInt8}, MTLi.contents(argbuf))
    _, maddr = Metal.malloc_buffer_and_gpu_address(dev)
    _, eaddr = Metal.exception_info_buffer_and_gpu_address(dev)
    state = Metal.KernelState(UInt32(1),
        reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, maddr),
        reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, eaddr),
        reinterpret(Core.LLVMPtr{UInt64, Metal.AS.Device}, UInt64(0)))
    function put(x, off)
        r = Ref(x)
        GC.@preserve r unsafe_copyto!(base + off,
            convert(Ptr{UInt8}, Base.unsafe_convert(Ptr{typeof(x)}, r)), sizeof(x))
    end
    put(state, 0)
    put(Metal.mtlconvert(range), 256)          # the range writer's arguments
    put(Metal.mtlconvert(flag), 512)
    put(UInt32(1), 768)                        # location: command 2, zero-based
    put(UInt32(ngated), 1024)                  # length, if the gate is open
    put(Metal.mtlconvert(out), 1280)           # the gated work's arguments
    put(1f0, 1536)

    desc = MTLi.MTLIndirectCommandBufferDescriptor(; max_kernel_buffers = 5)
    icb = MTLi.MTLIndirectCommandBuffer(dev, desc, 1 + ngated)
    c = MTLi.indirect_compute_command(icb, 1)
    MTLi.set_pipeline!(c, rngpip)
    MTLi.set_kernel_buffer!(c, argbuf, 0, 1)
    MTLi.set_kernel_buffer!(c, argbuf, 256, 2)
    MTLi.set_kernel_buffer!(c, argbuf, 512, 3)
    MTLi.set_kernel_buffer!(c, argbuf, 768, 4)
    MTLi.set_kernel_buffer!(c, argbuf, 1024, 5)
    MTLi.dispatch_threadgroups!(c, Metal.MTLSize(1), Metal.MTLSize(1))
    for i in 1:ngated
        g = MTLi.indirect_compute_command(icb, 1 + i)
        MTLi.set_pipeline!(g, addpip)
        MTLi.set_kernel_buffer!(g, argbuf, 0, 1)
        MTLi.set_kernel_buffer!(g, argbuf, 1280, 2)
        MTLi.set_kernel_buffer!(g, argbuf, 1536, 3)
        MTLi.dispatch_threadgroups!(g, Metal.MTLSize(1), Metal.MTLSize(64))
        MTLi.set_barrier!(g)
    end

    resources = (argbuf, mbuf, ebuf, out.data[], range.data[], flag.data[])
    function gated_replay!()
        cb = MTLi.MTLCommandBuffer(queue)
        enc = MTLi.MTLComputeCommandEncoder(cb)
        for r in resources
            MTLi.use!(enc, r, MTLi.ReadWriteUsage)
        end
        MTLi.execute_commands!(enc, icb, 1:1)                        # write the range
        MTLi.execute_commands_indirect!(enc, icb, range.data[], range.offset)
        MTLi.endEncoding!(enc)
        MTLi.commit!(cb)
        MTLi.wait_completed(cb)
        return nothing
    end

    # Gate closed: the same replay runs NOTHING, and the host said nothing about
    # it — the length the command processor read was written by a kernel one
    # command earlier.
    fill!(out, 0f0); fill!(flag, UInt32(0)); Metal.synchronize()
    gated_replay!()
    @test all(==(0f0), Array(out))
    @test Array(range) == UInt32[1, 0]

    # Gate open: four commands, serialised by their own barriers.
    fill!(flag, UInt32(1)); Metal.synchronize()
    gated_replay!()
    @test all(==(Float32(ngated)), Array(out))
    @test Array(range) == UInt32[1, ngated]

    GC.@preserve lib1 lib2 nothing
end

# A four-byte buffer an indirect command reaches by address is NOT made resident.
#
# `useResource` covers it for an ordinary dispatch — where the argument is packed
# bytes holding the same raw address, so nothing about how the kernel reaches the
# memory differs — and does not cover it for a command replayed out of an indirect
# buffer: the write below is dropped and a read of the same buffer comes back as
# zero, silently, exactly as an unmapped page does. Eight bytes is enough; four is
# not. Pinned rather than worked around because the failure has no symptom other
# than wrong numbers, and because a driver that fixes it should make this test say
# so rather than let the workaround stand unexplained.
#
# Mantle's pool never hands out one — its blocks are megabytes and a `GPURef` is a
# slice of one — so this is a hazard for hand-built buffers, which is what the
# recorder checks for at record time.
function icb_store!(a, v::UInt32)
    @inbounds a[1] = v
    return nothing
end

@testset "a four-byte resource is not resident for an indirect command" begin
    dev = Metal.device()
    queue = MTLi.MTLCommandQueue(dev)
    mbuf, _ = Metal.malloc_buffer_and_gpu_address(dev)
    ebuf, _ = Metal.exception_info_buffer_and_gpu_address(dev)
    pip, lib = icb_pipeline(icb_store!,
                            Tuple{Metal.MtlDeviceVector{UInt32,1}, UInt32}, "icb_store")

    function store_through_icb(n)
        a = Metal.MtlVector{UInt32}(undef, n); fill!(a, UInt32(9)); Metal.synchronize()
        argbuf = MTLi.MTLBuffer(dev, 1024; storage = Metal.SharedStorage)
        base = convert(Ptr{UInt8}, MTLi.contents(argbuf))
        _, maddr = Metal.malloc_buffer_and_gpu_address(dev)
        _, eaddr = Metal.exception_info_buffer_and_gpu_address(dev)
        state = Metal.KernelState(UInt32(1),
            reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, maddr),
            reinterpret(Core.LLVMPtr{UInt8, Metal.AS.Device}, eaddr),
            reinterpret(Core.LLVMPtr{UInt64, Metal.AS.Device}, UInt64(0)))
        function put(x, off)
            r = Ref(x)
            GC.@preserve r unsafe_copyto!(base + off,
                convert(Ptr{UInt8}, Base.unsafe_convert(Ptr{typeof(x)}, r)), sizeof(x))
        end
        put(state, 0); put(Metal.mtlconvert(a), 256); put(UInt32(42), 512)
        desc = MTLi.MTLIndirectCommandBufferDescriptor(; max_kernel_buffers = 3)
        icb = MTLi.MTLIndirectCommandBuffer(dev, desc, 1)
        c = MTLi.indirect_compute_command(icb, 1)
        MTLi.set_pipeline!(c, pip)
        for (i, off) in enumerate((0, 256, 512))
            MTLi.set_kernel_buffer!(c, argbuf, off, i)
        end
        MTLi.dispatch_threadgroups!(c, Metal.MTLSize(1), Metal.MTLSize(1))
        cb = MTLi.MTLCommandBuffer(queue)
        enc = MTLi.MTLComputeCommandEncoder(cb)
        for r in (argbuf, mbuf, ebuf, a.data[])
            MTLi.use!(enc, r, MTLi.ReadWriteUsage)
        end
        MTLi.execute_commands!(enc, icb, 1:1)
        MTLi.endEncoding!(enc)
        MTLi.commit!(cb)
        MTLi.wait_completed(cb)
        return Array(a)[1]
    end

    @test store_through_icb(1) == UInt32(9)      # dropped: four bytes
    @test store_through_icb(2) == UInt32(42)     # landed: eight

    # The same four-byte buffer, from an ordinary launch, takes the store — so the
    # buffer is fine and the indirect path is what loses it.
    a = Metal.MtlVector{UInt32}(undef, 1); fill!(a, UInt32(9)); Metal.synchronize()
    Metal.@metal threads=1 groups=1 icb_store!(Metal.mtlconvert(a), UInt32(42))
    Metal.synchronize()
    @test Array(a)[1] == UInt32(42)

    GC.@preserve lib nothing
end
