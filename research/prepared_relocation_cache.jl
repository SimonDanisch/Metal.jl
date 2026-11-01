# Persistent-cache boundary probe, not a production cache policy.
# Uses only portable symbol/constant-binding descriptors and the freshly lowered
# relocation manifest. Never loads an old manifest or process address from disk.
# Select JULIA_OBJCACHE_PATH before loading Metal for an isolated cache; after
# restarting Julia, set METAL_RELOCATION_PROBE_CACHED=1 to require cache hits.
# Unsupported targets throw here; a production implementation must take a miss.
# This does not skip Julia inference, LLVM generation or relocation preparation.

module PreparedRelocationCacheProbe
using Metal, LLVM, Test, SHA
const GC = Metal.GPUCompiler
const NS = "Metal/experimental-prepared-relocations-20261101"

function kernel!(out, input, ::Val{Bias}) where {Bias}
    i = Metal.thread_position_in_grid_1d()
    @inbounds out[i] = input[i] + UInt(pointer_from_objref(:prepared_cache_probe)) + (UInt(pointer_from_objref(Float32)) << 1) + UInt(Bias)
    return
end

function target_descriptor(target)
    if target isa GC.JuliaValueRef
        v = target.value
        v isa Symbol && return ("symbol", String(v))
        if v isa Function || (v isa DataType && isempty(v.parameters))
            owner = parentmodule(v)
            name = nameof(v)
            if isdefined(owner, name) && isconst(owner, name) && getfield(owner, name) === v
                return ("binding", string(Base.PkgId(owner)), join(Base.fullname(owner), '.'), String(name))
            end
        end
    elseif target isa GC.CGlobalRef
        return ("cglobal", String(target.symbol), target.library, target.offset)
    end
    error("Unsupported portable target: ", repr(target))
end

function compile(job)
    return GC.JuliaContext() do _
        mod, meta = GC.compile(:llvm, job)
        try
            job.config.strip && LLVM.strip_debuginfo!(mod)
            GC.prepare_execution!(job, mod, meta.relocations)
            targets = map(target_descriptor, meta.relocations.table)
            @test length(targets) == 2
            @test isempty(meta.relocations.resolved)
            @test meta.relocations.frozen[]
            key = Metal.ObjCache.keyhash(1, Metal.canonical_ir(mod), Metal.compiler_identity(job), repr(targets))
            reversed_key = Metal.ObjCache.keyhash(1, Metal.canonical_ir(mod), Metal.compiler_identity(job), repr(reverse(targets)))
            @test key != reversed_key
            found = Metal.ObjCache.get(NS, key)
            artifacts = if isnothing(found)
                loggingEnabled = haskey(mod.functions, "air.os_log")
                air = GC.mcgen(job, mod, LLVM.CodeGenFileType.Object)
                entry = meta.entry.name
                fn = Metal.MetalLibFunction(;
                    name = entry, air_module = air,
                    air_version = job.config.target.air, metal_version = job.config.target.metal
                )
                lib = Metal.MetalLib(;
                    functions = [fn], file_version = Metal.metallib_target(job.config.target.macos),
                    platform_version = job.config.target.macos, uuid = Metal.content_uuid(air)
                )
                io = IOBuffer();write(io, lib);metallib = take!(io)
                a = (; air, metallib, entry, loggingEnabled)
                @test Metal.ObjCache.put!(NS, key, Metal.persisted_bytes(a))
                a
            else
                Metal.persisted_artifacts(found)
            end
            # No addresses or old manifest are loaded. The current preparation fixes
            # the exact table order used by both the IR key and this session's loader.
            return (; artifacts..., relocations = meta.relocations), (; key = bytes2hex(key), hit = !isnothing(found), targets)
        finally
            LLVM.dispose(mod)
        end
    end
end

function run(; expect_hit = nothing, bias = 0)
    dev = Metal.device()
    input = Metal.MtlArray(UInt[1, 2, 3, 4])
    output = similar(input)
    try
        tt = Tuple{typeof(Metal.mtlconvert(output)), typeof(Metal.mtlconvert(input)), typeof(Val(bias))}
        cfg = Metal.compiler_config(dev; name = "prepared_relocation_cache_probe")
        job = GC.CompilerJob(Metal.methodinstance(typeof(kernel!), tt), cfg)
        artifacts, stats = compile(job)
        pipeline = Metal.link_pipeline(dev, artifacts.air, artifacts.metallib, artifacts.entry, nothing)
        table = Metal.reloc_table_buffer(dev, artifacts.relocations)
        kernel = Metal.HostKernel{typeof(kernel!), typeof(kernel!), tt}(
            kernel!, kernel!, pipeline, artifacts.loggingEnabled, dev,
            Int(pipeline.maxTotalThreadsPerThreadgroup), Metal.MTL.MTLSize(0, 0, 0),
            Int(pipeline.staticThreadgroupMemoryLength), Int(pipeline.threadExecutionWidth),
            Metal.can_use_residency_sets(dev), table
        )
        kernel(output, input, Val(bias); threads = 4)
        Metal.synchronize()
        actual = Array(output)
        @test actual == UInt[1, 2, 3, 4] .+ UInt(pointer_from_objref(:prepared_cache_probe)) .+ (UInt(pointer_from_objref(Float32)) << 1) .+ UInt(bias)
        expect_hit === nothing || @test stats.hit === expect_hit
        println(stats)
        return stats
    finally
        Metal.unsafe_free!(input);Metal.unsafe_free!(output)
    end
end
end
using Test
@testset "Prepared relocation cache uses current bindings" begin
    expected_hit = get(ENV, "METAL_RELOCATION_PROBE_CACHED", "0") == "1" ? true : nothing
    a = PreparedRelocationCacheProbe.run(; expect_hit = expected_hit)
    b = PreparedRelocationCacheProbe.run(; bias = 1, expect_hit = expected_hit)
    @test a.key != b.key
    c = PreparedRelocationCacheProbe.run(; expect_hit = true)
    @test c.key == a.key
    @test_throws ErrorException PreparedRelocationCacheProbe.target_descriptor(PreparedRelocationCacheProbe.GC.JuliaValueRef(Ref(1)))
end
