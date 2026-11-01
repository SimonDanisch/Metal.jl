# Native binary-function reuse through Metal's visible function tables.
# Run with include("research/binary_function_reuse.jl") in a Metal environment.
# This is an MSL feature probe, not a Julia helper cache or a renderer benchmark.
# https://developer.apple.com/videos/play/wwdc2020/10013/

module BinaryFunctionReuseProbe

using Metal, Test
using Metal.MTL: @objc, id, Out, NSError, NSArray, MTLLibrary, MTLFunction,
    MTLFunctionDescriptor, MTLLinkedFunctions, MTLComputePipelineState,
    MTLVisibleFunctionTable, MTLVisibleFunctionTableDescriptor, MTLFunctionHandle,
    MTLComputeCommandEncoder

const MTL = Metal.MTL

function binary_function(lib, name)
    desc = MTLFunctionDescriptor()
    desc.name = name
    desc.options = MTL.MTLFunctionOptionCompileToBinary
    err = Out{NSError}()
    fun = @objc [lib::id{MTLLibrary} newFunctionWithDescriptor:desc::id{MTLFunctionDescriptor}
        error:err::Out{NSError}]::Union{Nothing, MTLFunction}
    isnothing(fun) && MTL.throw_error(err)
    return fun
end

function run()
    dev = Metal.device()
    source = """
    #include <metal_stdlib>
    #include <metal_visible_function_table>
    using namespace metal;
    using Op = float(float);
    [[visible]] float squared(float x) { return x*x; }
    [[visible]] float doubled(float x) { return 2.0f*x; }
    kernel void one_op(device const float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       visible_function_table<Op> table [[buffer(2)]],
                       uint tid [[thread_position_in_grid]]) {
        output[tid] = table[tid & 1](input[tid]);
    }
    kernel void composed_ops(device const float* input [[buffer(0)]],
                             device float* output [[buffer(1)]],
                             visible_function_table<Op> table [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
        output[tid] = table[(tid + 1) & 1](table[tid & 1](input[tid]));
    }
    """
    lib = MTL.MTLLibrary(dev, source)
    # Compile each helper once, then pass these same objects to both pipelines.
    funcs = [binary_function(lib, "squared"), binary_function(lib, "doubled")]
    linked = @objc [MTLLinkedFunctions new]::MTLLinkedFunctions
    linked.binaryFunctions = NSArray(funcs)
    xs = collect(range(0.1f0, 0.9f0; length = 256))
    input = Metal.MtlArray(xs)
    output = Metal.MtlArray(zeros(Float32, length(xs)))
    queue = MTL.MTLCommandQueue(dev)
    rows = NamedTuple[]
    try
        Metal.synchronize()
        for name in ("one_op", "composed_ops")
            desc = MTL.MTLComputePipelineDescriptor()
            desc.computeFunction = MTLFunction(lib, name)
            desc.linkedFunctions = linked
            pipeline = MTLComputePipelineState(dev, desc)
            tabledesc = @objc [MTLVisibleFunctionTableDescriptor new]::MTLVisibleFunctionTableDescriptor
            tabledesc.functionCount = length(funcs)
            table = @objc [pipeline::id{MTLComputePipelineState} newVisibleFunctionTableWithDescriptor:tabledesc::id{MTLVisibleFunctionTableDescriptor}]::MTLVisibleFunctionTable
            for (i, fun) in enumerate(funcs)
                handle = @objc [pipeline::id{MTLComputePipelineState} functionHandleWithFunction:fun::id{MTLFunction}]::MTLFunctionHandle
                @objc [table::id{MTLVisibleFunctionTable} setFunction:handle::id{MTLFunctionHandle}
                    atIndex:(i - 1)::UInt64]::Nothing
            end
            cb = MTL.MTLCommandBuffer(queue)
            MTLComputeCommandEncoder(cb) do enc
                MTL.set_function!(enc, pipeline)
                MTL.set_buffer!(enc, input.data[].buffer, input.offset, 1)
                MTL.set_buffer!(enc, output.data[].buffer, output.offset, 2)
                @objc [enc::id{MTLComputeCommandEncoder} setVisibleFunctionTable:table::id{MTLVisibleFunctionTable}
                    atBufferIndex:2::UInt64]::Nothing
                MTL.dispatchThreads!(enc, MTL.MTLSize(length(xs), 1, 1), MTL.MTLSize(64, 1, 1))
            end
            MTL.commit!(cb)
            MTL.wait_completed(cb)
            expected = [isodd(i) ? x*x : 2.0f0*x for (i, x) in enumerate(xs)]
            if name == "composed_ops"
                expected = [isodd(i) ? 2.0f0*x : x*x for (i, x) in enumerate(expected)]
            end
            actual = Array(output)
            @test actual == expected
            @test all(isfinite, actual)
            push!(rows, (; name, max_error = maximum(abs.(actual .- expected))))
        end
    finally
        Metal.unsafe_free!(input)
        Metal.unsafe_free!(output)
    end
    return rows
end

end

using Test
@testset "Metal native binary functions reused in distinct kernels" begin
    println(BinaryFunctionReuseProbe.run())
end
