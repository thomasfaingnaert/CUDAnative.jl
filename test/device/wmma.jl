@testset "WMMA" begin

################################################################################

    @testset "LLVM intrinsics" begin

        @testset "wmma_store_d" begin
            buf     = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float32))
            buf_ptr = convert(CuPtr{Float32}, buf)

            function kernel(buf_ptr)
                data = (42, 42, 42, 42, 42, 42, 42, 42)
                wmma_store_d(buf_ptr, data..., 16)
                return
            end

            @cuda threads=32 kernel(buf_ptr)

            res = zeros(Float32, 16 * 16)
            unsafe_copyto!(pointer(res), buf_ptr, 16 * 16)

            @test all(res .== 42.0)
        end

        @testset "wmma_load" begin
            buf     = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float16))
            buf_ptr = convert(CuPtr{Float16}, buf)

            res = 42 * ones(Float16, 16 * 16)
            unsafe_copyto!(buf_ptr, pointer(res), 16 * 16)

            output = Mem.alloc(Mem.Device, sizeof(Bool))
            output_ptr = convert(CuPtr{Bool}, output)

            function kernel(buf_ptr, output_ptr)
                data_a = wmma_load_a(buf_ptr, 16)
                data_b = wmma_load_b(buf_ptr, 16)

                data_ok = data -> all(val -> val == (VecElement{Float16}(42), VecElement{Float16}(42)), data)

                if data_ok(data_a) && data_ok(data_b)
                    unsafe_store!(output_ptr, 1)
                end

                return
            end

            @cuda threads=32 kernel(buf_ptr, output_ptr)

            result = zeros(Bool, 1)
            unsafe_copyto!(pointer(result), output_ptr, 1)

            @test all(result)
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
