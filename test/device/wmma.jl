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
            # Matmul kernel
            function kernel(res, a, b)
                a_frag = wmma_load_a(a, 16)
                b_frag = wmma_load_b(b, 16)

                res_frag = wmma_mma(a_frag..., b_frag...)

                wmma_store_d(res, res_frag..., 16)
                return
            end

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

            # Generate random input
            a = rand(Float16, 16 * 16)
            b = rand(Float16, 16 * 16)

            # Allocate memory on GPU
            dev_res     = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float32))
            dev_res_ptr = convert(CuPtr{Float32}, dev_res)

            dev_a       = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float16))
            dev_a_ptr   = convert(CuPtr{Float16}, dev_a)

            dev_b       = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float16))
            dev_b_ptr   = convert(CuPtr{Float16}, dev_b)

            # Copy input to the GPU
            unsafe_copyto!(dev_a_ptr, pointer(a), 16 * 16)
            unsafe_copyto!(dev_b_ptr, pointer(b), 16 * 16)

            # Perform multiplication
            @cuda threads=32 kernel(dev_res_ptr, dev_a_ptr, dev_b_ptr)

            # Check result
            res = zeros(Float32, 16 * 16)
            unsafe_copyto!(pointer(res), dev_res_ptr, 16 * 16)
            @test check_matrix_mul(a, b, res)
        end
    end

################################################################################

end
