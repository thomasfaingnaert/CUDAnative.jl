@testset "WMMA" begin

################################################################################

    @testset "LLVM intrinsics" begin
        @testset "wmma_store_d" begin
            output     = Array{Float32}(undef, (16, 16))
            output_dev = CuArray(output)

            function kernel(output_dev)
                data = (42, 42, 42, 42, 42, 42, 42, 42)
                wmma_store_d(pointer(output_dev), data, 16)
                return
            end

            @cuda threads=32 kernel(output_dev)

            @test all(Array(output_dev) .== 42.0)
        end

        @testset "wmma_load" begin
            input      = 42 * ones(Float16, (16, 16))
            input_dev  = CuArray(input)
            result     = Array{Bool}(undef, 1)
            result_dev = CuArray(result)

            function kernel(input_dev, result_dev)
                data_a = wmma_load_a(pointer(input_dev), 16)
                data_b = wmma_load_b(pointer(input_dev), 16)

                data_ok = data -> all(val -> val == Float16(42), data)
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
            d     = Array{Float32}(undef, (16, 16))
            d_dev = CuArray(d)

            # Matrix multiply kernel (D = A * B)
            function kernel(a_dev, b_dev, d_dev)
                a_frag = wmma_load_a(pointer(a_dev), 16)
                b_frag = wmma_load_b(pointer(b_dev), 16)
                d_frag = wmma_mma(a_frag, b_frag)
                wmma_store_d(pointer(d_dev), d_frag, 16)

                return
            end

            # Matrix multiply check on CPU
            function check_matrix_mul(a, b, res)
                for i = 1:16, j = 1:16
                    if res[i, j] ≉ sum(a[i, 1:16] .* b[1:16, j]) rtol=0.01
                        return false
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

    @testset "CUDA C-style interface" begin
        @testset "Matrix multiplication" begin
            # Generate input matrices
            a     = rand(Float16, (16, 16))
            a_dev = CuArray(a)
            b     = rand(Float16, (16, 16))
            b_dev = CuArray(b)

            # Reserve space for result
            d     = Array{Float32}(undef, (16, 16))
            d_dev = CuArray(d)

            # Matrix multiply kernel (D = A * B), using higher-level API
            function kernel(a_dev, b_dev, d_dev)
                # Declare the fragments
                a_frag = wmma_fragment(wmma_matrix_a, 16, 16, 16, Float16, wmma_col_major)
                b_frag = wmma_fragment(wmma_matrix_b, 16, 16, 16, Float16, wmma_col_major)
                c_frag = wmma_fragment(wmma_matrix_accumulator, 16, 16, 16, Float32)

                # Initialise the accumulator to zero
                wmma_fill_fragment(c_frag, 0.0f0)

                # Load inputs from memory
                wmma_load_matrix_sync(a_frag, pointer(a_dev), 16)
                wmma_load_matrix_sync(b_frag, pointer(b_dev), 16)

                # Perform matrix multiply
                wmma_mma_sync(c_frag, a_frag, b_frag, c_frag)

                # Store the output
                wmma_store_matrix_sync(pointer(d_dev), c_frag, 16)

                return
            end

            # Matrix multiply check on CPU
            function check_matrix_mul(a, b, res)
                for i = 1:16, j = 1:16
                    if res[i, j] ≉ sum(a[i, 1:16] .* b[1:16, j]) rtol=0.01
                        return false
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
end
