using LinearAlgebra

@testset "mixed-precision matrix matrix multiplication" begin
    N = 10
    rows_a = N
    cols_a = N

    rows_b = N
    cols_b = N

    rows_c = rows_a
    cols_c = cols_b

    alpha = Float64(1)
    beta  = Float64(1)

    for (input_jl_type, accum_jl_type) in MPS.MPS_VALID_MATMUL_TYPES
        @testset let input_jl_type = input_jl_type, accum_jl_type = accum_jl_type
            arr_a = rand(input_jl_type, (rows_a, cols_a))
            arr_b = rand(input_jl_type, (rows_b, cols_b))
            arr_c = zeros(accum_jl_type, (rows_c, cols_c))

            buf_a = MtlArray{input_jl_type}(arr_a)
            buf_b = MtlArray{input_jl_type}(arr_b)
            buf_c = Metal.zeros(accum_jl_type, size(arr_c))

            truth_c = (alpha .* accum_jl_type.(arr_a)) * accum_jl_type.(arr_b) .+ (beta .* arr_c)

            MPS.matmul!(buf_c, buf_a, buf_b, alpha, beta)

            @test Array(buf_c) ≈ truth_c
        end
    end
end

@testset "batched matrix matrix multiplication" begin
    M = 8
    N = 7
    P = 9
    batch_size = 3

    rows_a = M
    cols_a = N

    rows_b = N
    cols_b = P

    rows_c = M
    cols_c = P

    alpha = Float64(1)
    beta = Float64(1)

    for (input_jl_type, accum_jl_type) in MPS.MPS_VALID_MATMUL_TYPES
        @testset let input_jl_type = input_jl_type, accum_jl_type = accum_jl_type
            arr_a = rand(input_jl_type, (rows_a, cols_a, batch_size))
            arr_b = rand(input_jl_type, (rows_b, cols_b, batch_size))
            arr_c = zeros(accum_jl_type, (rows_c, cols_c, batch_size))

            buf_a = MtlArray{input_jl_type}(arr_a)
            buf_b = MtlArray{input_jl_type}(arr_b)
            buf_c = Metal.zeros(accum_jl_type, (rows_c, cols_c, batch_size))

            truth_c = zeros(accum_jl_type, (rows_c, cols_c, batch_size))
            for i in 1:batch_size
                @views truth_c[:, :, i] = (alpha .* accum_jl_type.(arr_a[:, :, i])) * accum_jl_type.(arr_b[:, :, i]) .+ (beta .* arr_c[:, :, i])
            end

            MPS.matmul!(buf_c, buf_a, buf_b, alpha, beta)

            @test Array(buf_c) ≈ truth_c
        end
    end
end

@testset "mixed-precision matrix vector multiplication" begin
    N = 10
    rows = N
    cols = N

    alpha = Float64(1)
    beta  = Float64(0)

    @testset "$(input_jl_type) => $accum_jl_type" for (input_jl_type, accum_jl_type) in MPS.MPS_VALID_MATVECMUL_TYPES
        arr_a = rand(input_jl_type, (rows,cols))
        arr_b = rand(input_jl_type, (rows,))
        arr_c = zeros(accum_jl_type, (rows,))

        buf_a = MtlArray{input_jl_type}(arr_a)
        buf_b = MtlArray{input_jl_type}(arr_b)
        buf_c = Metal.zeros(accum_jl_type, (rows,))

        truth_c = (alpha .* accum_jl_type.(arr_a)) *  accum_jl_type.(arr_b) .+ (beta .* arr_c)

        MPS.matvecmul!(buf_c, buf_a, buf_b, alpha, beta)

        @test Array(buf_c) ≈ truth_c
    end
end

@testset "MPS linear solvers" begin
    T = Float32
    n = 32
    nrhs = 3

    A = rand(T, n, n)
    A .+= T(n) .* Matrix{T}(I, n, n)
    b = rand(T, n)
    B = rand(T, n, nrhs)

    dA = MtlMatrix(A)
    db = MtlVector(b)
    dB = MtlMatrix(B)

    x = MPS.solve_lu(dA, db)
    X = MPS.solve_lu(dA, dB)
    @test Array(x) ≈ A \ b rtol=1f-4
    @test Array(X) ≈ A \ B rtol=1f-4
    @test Array(dA * x) ≈ b rtol=1f-4
    @test Array(dA * X) ≈ B rtol=1f-4

    F = lu(dA)
    xreuse = MPS.solve_lu(F, db; out=copy(db))
    Xreuse = MPS.solve_lu(F, dB; out=copy(dB))
    @test Array(xreuse) ≈ A \ b rtol=1f-4
    @test Array(Xreuse) ≈ A \ B rtol=1f-4

    M = rand(T, n, n)
    SPD = M'M + T(n) .* Matrix{T}(I, n, n)
    dSPD = MtlMatrix(SPD)
    xc = MPS.solve_cholesky(dSPD, db)
    Xc = MPS.solve_cholesky(dSPD, dB)
    @test Array(xc) ≈ SPD \ b rtol=1f-4
    @test Array(Xc) ≈ SPD \ B rtol=1f-4

    C = cholesky(Symmetric(dSPD, :U))
    xcreuse = MPS.solve_cholesky(C, db; out=copy(db))
    Xcreuse = MPS.solve_cholesky(C, dB; out=copy(dB))
    @test Array(xcreuse) ≈ SPD \ b rtol=1f-4
    @test Array(Xcreuse) ≈ SPD \ B rtol=1f-4

    U = triu(A)
    L = tril(A)
    BR = rand(T, nrhs, n)
    dU = MtlMatrix(U)
    dL = MtlMatrix(L)
    dBR = MtlMatrix(BR)
    @test Array(MPS.solve_triangular(dU, dB; upper=true, unit=false, out=copy(dB))) ≈
          UpperTriangular(U) \ B rtol=1f-4
    @test Array(MPS.solve_triangular(dL, dB; upper=false, unit=false, out=copy(dB))) ≈
          LowerTriangular(L) \ B rtol=1f-4
    @test Array(MPS.solve_triangular(dU, dB; upper=true, unit=false, transpose=true,
                                     out=copy(dB))) ≈
          transpose(UpperTriangular(U)) \ B rtol=1f-4
    @test Array(MPS.solve_triangular(dU, dBR; upper=true, unit=false, right=true,
                                     out=copy(dBR))) ≈
          BR / UpperTriangular(U) rtol=1f-4

    # Regression coverage for JuliaGPU/Metal.jl#145: older MPS solve code reported
    # blocks of NaNs for systems larger than 128x128.
    nlarge = 160
    Alarge = rand(T, nlarge, nlarge)
    Alarge .+= T(nlarge) .* Matrix{T}(I, nlarge, nlarge)
    Blarge = rand(T, nlarge, nlarge)
    dAlarge = MtlMatrix(Alarge)
    dBlarge = MtlMatrix(Blarge)

    Xlarge = Array(MPS.solve_lu(dAlarge, dBlarge))
    @test all(isfinite, Xlarge)
    @test Xlarge ≈ Alarge \ Blarge rtol=1f-4

    Flarge = lu(dAlarge)
    Xlarge = Array(MPS.solve_lu(Flarge, dBlarge; out=copy(dBlarge)))
    @test all(isfinite, Xlarge)
    @test Xlarge ≈ Alarge \ Blarge rtol=1f-4

    Ularge = triu(Alarge)
    Xlarge = Array(MPS.solve_triangular(MtlMatrix(Ularge), dBlarge;
                                        upper=true, unit=false, out=copy(dBlarge)))
    @test all(isfinite, Xlarge)
    @test Xlarge ≈ UpperTriangular(Ularge) \ Blarge rtol=1f-4

    Llarge = tril(Alarge)
    Xlarge = Array(MPS.solve_triangular(MtlMatrix(Llarge), dBlarge;
                                        upper=false, unit=false, out=copy(dBlarge)))
    @test all(isfinite, Xlarge)
    @test Xlarge ≈ LowerTriangular(Llarge) \ Blarge rtol=1f-4
end

@testset "topk & topk!" begin
    # Modified from https://github.com/FluxML/NNlib.jl/pull/353
    function cpu_topk(x::Matrix{T}, k; rev=true, dims=1) where {T}
        if dims === nothing
            y = vec(x)
            perm = partialsortperm(y, 1:k; rev)
            return CartesianIndices(x)[perm], y[perm]
        else
            @assert dims isa Int
            sz1 = size(x)[1:dims-1]
            sz2 = size(x)[dims+1:end]
            slice1 = CartesianIndices(sz1)
            slice2 = CartesianIndices(sz2)
            perm = similar(x, Int, (sz1..., k, sz2...))
            y = similar(x, (sz1..., k, sz2...))
            for I1 in slice1
                for I2 in slice2
                    xI = x[I1,:,I2]
                    permI = partialsortperm(x[I1,:,I2], 1:k; rev)
                    perm[I1,:,I2] .= permI
                    y[I1,:,I2] .= xI[permI]
                end
            end
            return perm, y
        end
    end
    # WHICH index a tie resolves to is not part of the contract, and `rand(Float16, 20,
    # 30)` has a tie inside its top five about half the time — eleven mantissa bits over
    # twenty draws — so comparing indices to `partialsortperm`'s choice failed roughly
    # every other run, on an unseeded input, for no reason to do with the kernel. What IS
    # well defined is the values (a tie does not change the multiset) and that each index
    # points at the value beside it. Both are checked; the index comparison is kept for the
    # data where it means something.
    @testset "$ftype" for ftype in (Float16, Float32)
        # Normal operation
        for (shp,k) in [((3,1), 2), ((20,30), 5)]
            cpu_a = rand(ftype, shp...)
            cpu_i, cpu_v = cpu_topk(cpu_a, k)
            # A tie anywhere inside the top `k` of a column makes the indices ambiguous.
            decidable = all(1:shp[2]) do c
                col = sort(cpu_a[:, c]; rev = true)
                all(col[j] != col[j+1] for j in 1:min(k, length(col) - 1))
            end
            a = MtlMatrix(cpu_a)

            for (i, v) in (MPS.topk(a, k),
                           #topk!
                           MPS.topk!(a, MtlMatrix{UInt32}(undef, (k, shp[2])),
                                     MtlMatrix{ftype}(undef, (k, shp[2])), k))
                hi, hv = Array(i), Array(v)
                @test hv == cpu_v
                # Every index names the value returned with it, which is the contract a
                # tie cannot make ambiguous.
                @test all(cpu_a[hi[j, c], c] == hv[j, c]
                          for j in 1:k, c in 1:shp[2])
                decidable && @test hi == cpu_i
            end
        end
        shp = (20,30)
        k = 17

        cpu_a = rand(ftype, shp...)
        cpu_i, cpu_v = cpu_topk(cpu_a, k)

        a = MtlMatrix(cpu_a)
        @test_throws "MPSMatrixFindTopK does not support values of k > 16" i, v = MPS.topk(a, k)

        #topk!
        i = MtlMatrix{UInt32}(undef, (k, shp[2]))
        v = MtlMatrix{ftype}(undef, (k, shp[2]))

        @test_throws "MPSMatrixFindTopK does not support values of k > 16" i, v = MPS.topk!(a, i, v, k)
    end
end

using .MPS: MPSMatrixSoftMax, MPSMatrixLogSoftMax
@testset "MPSMatrixSoftMax" begin
    # NOT `rand(Int)`: `sourceColumns` is an `NSUInteger`, and a negative `Int64` assigned
    # to it throws `InexactError` — which `rand(Int)` produces about half the time, on an
    # unseeded draw, so this testset errored out roughly every other run. A row or column
    # count is non-negative by construction; the point is that the property round-trips.
    cols = rand(1:typemax(Int32))
    rows = rand(1:typemax(Int32))

    skern = MPSMatrixSoftMax(device())
    skern.sourceColumns = cols
    skern.sourceRows = rows

    @test skern isa MPSMatrixSoftMax
    @test skern.sourceColumns == cols
    @test skern.sourceRows == rows

    lkern = MPSMatrixLogSoftMax(device())
    lkern.sourceColumns = cols
    lkern.sourceRows = rows

    @test lkern isa MPSMatrixLogSoftMax
    @test lkern.sourceColumns == cols
    @test lkern.sourceRows == rows
end
