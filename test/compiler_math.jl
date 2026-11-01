using Test, Metal

@noinline compiler_clamped_sqrt(x::Float32) = sqrt(max(x, 0.0f0))

function compiler_clamped_sqrt_kernel!(out, input)
    i = Metal.thread_position_in_grid().x
    if i <= length(input)
        @inbounds out[i] = compiler_clamped_sqrt(input[i])
    end
    return
end

@testset "optimizable floating-point bounds" begin
    ir = sprint(io -> Metal.code_llvm(io, compiler_clamped_sqrt, Tuple{Float32}; debuginfo = :none))
    @test occursin("llvm.maxnum.f32", ir)
    @test !occursin("throw_complex_domainerror", ir)
    @test !occursin("llvm.trap", ir)

    # A bounded helper can remain out of line: no domain-error exit needs
    # hoisting into the kernel during AIR lowering.
    air = sprint(
        io -> Metal.code_air(
            io, compiler_clamped_sqrt_kernel!,
            Tuple{Metal.MtlDeviceVector{Float32, Metal.AS.Device}, Metal.MtlDeviceVector{Float32, Metal.AS.Device}};
            kernel = true, dump_module = true
        )
    )
    @test occursin(r"define[^\n]*@julia_compiler_clamped_sqrt_", air)

    inputs = Float32[-Inf, -4, -0.0, 0, 4, Inf, NaN]
    input = MtlArray(inputs);output = similar(input)
    Metal.@sync @metal threads = length(inputs) compiler_clamped_sqrt_kernel!(output, input)
    expected = [isnan(x) ? 0.0f0 : sqrt(max(x, 0.0f0)) for x in inputs]
    @test Array(output) == expected
end
