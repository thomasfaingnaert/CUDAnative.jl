@testset "WMMA" begin

################################################################################

    @testset "LLVM intrinsics" begin

        @testset "wmma_store_d" begin
            output     = Array{Float32, 2}(undef, (16, 16))
            output_dev = CuArray(output)

            function kernel(output_dev)
                data = (42, 42, 42, 42, 42, 42, 42, 42)
                wmma_store_d(pointer(output_dev), data..., 16)
                return
            end

            @cuda threads=32 kernel(output_dev)

            @test all(Array(output_dev) .== 42.0)
        end

        @testset "wmma_load" begin
            input      = 42 * ones(Float16, (16, 16))
            input_dev  = CuArray(input)
            result     = Array{Bool, 1}(undef, 1)
            result_dev = CuArray(result)

            function kernel(input_dev, result_dev)
                data_a = wmma_load_a(pointer(input_dev), 16)
                data_b = wmma_load_b(pointer(input_dev), 16)

                data_ok = data -> all(val -> val == (VecElement{Float16}(42), VecElement{Float16}(42)), data)
                result_dev[1] = data_ok(data_a) && data_ok(data_b)

                return
            end

            @cuda threads=32 kernel(input_dev, result_dev)

            @test all(Array(result_dev))
        end

        @testset "wmma_mma" begin
            # Generate input matrices
            a     = rand(Float16, (16, 16))
            a_dev = CuArray(a)
            b     = rand(Float16, (16, 16))
            b_dev = CuArray(b)

            # Reserve space for result
            d     = Array{Float32, 2}(undef, (16, 16))
            d_dev = CuArray(d)

            # Matrix multiply kernel (D = A * B)
            function kernel(a_dev, b_dev, d_dev)
                a_frag = wmma_load_a(pointer(a_dev), 16)
                b_frag = wmma_load_b(pointer(b_dev), 16)
                d_frag = wmma_mma(a_frag..., b_frag...)
                wmma_store_d(pointer(d_dev), d_frag..., 16)

                return
            end

            # Matrix multiply check on CPU
            function check_matrix_mul(a, b, res)
                for i = 0:15
                    for j = 0:15
                        tmp = 0

                        for k = 0:15
                            tmp += a[1 + 16 * i + k] * b[1 + 16 * k + j]
                        end

                        if tmp ≉ res[1 + 16 * i + j] rtol=0.01
                            return false
                        end
                    end
                end

                return true
            end

            # Perform multiply
            @cuda threads=32 kernel(a_dev, b_dev, d_dev)

            # Check the result
            @test check_matrix_mul(a, b, Array(d_dev))
        end
    end

################################################################################

end
