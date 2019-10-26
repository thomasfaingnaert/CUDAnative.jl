@testset "WMMA" begin

################################################################################

    @testset "LLVM intrinsics" begin

        @testset "wmma_store_d" begin
            buf     = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float32))
            buf_ptr = convert(CuPtr{Float32}, buf)

            function kernel(buf_ptr)
                data = (42, 42, 42, 42, 42, 42, 42, 42)
                CUDAnative.wmma_store_d(buf_ptr, data..., 16)
                return
            end

            @cuda threads=32 kernel(buf_ptr)

            res = zeros(Float32, 16 * 16)
            unsafe_copyto!(pointer(res), buf_ptr, 16 * 16)

            @test all(res .== 42.0)
        end

        @testset "wmma_load_a" begin
            buf     = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float16))
            buf_ptr = convert(CuPtr{Float16}, buf)

            res = 42 * ones(Float16, 16 * 16)
            unsafe_copyto!(buf_ptr, pointer(res), 16 * 16)

            output = Mem.alloc(Mem.Device, sizeof(Bool))
            output_ptr = convert(CuPtr{Bool}, output)

            function kernel(buf_ptr, output_ptr)
                data = CUDAnative.wmma_load_a(buf_ptr, 16)

                expected = (VecElement{Float16}(42), VecElement{Float16}(42))

                if data.data0 != expected return end
                if data.data1 != expected return end
                if data.data2 != expected return end
                if data.data3 != expected return end
                if data.data4 != expected return end
                if data.data5 != expected return end
                if data.data6 != expected return end
                if data.data7 != expected return end

                unsafe_store!(output_ptr, 1)
                return
            end

            @cuda threads=32 kernel(buf_ptr, output_ptr)

            result = zeros(Bool, 1)
            unsafe_copyto!(pointer(result), output_ptr, 1)

            @test all(result)
        end

        @testset "wmma_load_b" begin
            buf     = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float16))
            buf_ptr = convert(CuPtr{Float16}, buf)

            res = 42 * ones(Float16, 16 * 16)
            unsafe_copyto!(buf_ptr, pointer(res), 16 * 16)

            output = Mem.alloc(Mem.Device, sizeof(Bool))
            output_ptr = convert(CuPtr{Bool}, output)

            function kernel(buf_ptr, output_ptr)
                data = CUDAnative.wmma_load_b(buf_ptr, 16)

                expected = (VecElement{Float16}(42), VecElement{Float16}(42))

                if data.data0 != expected return end
                if data.data1 != expected return end
                if data.data2 != expected return end
                if data.data3 != expected return end
                if data.data4 != expected return end
                if data.data5 != expected return end
                if data.data6 != expected return end
                if data.data7 != expected return end

                unsafe_store!(output_ptr, 1)
                return
            end

            @cuda threads=32 kernel(buf_ptr, output_ptr)

            result = zeros(Bool, 1)
            unsafe_copyto!(pointer(result), output_ptr, 1)

            @test all(result)
        end

    end

################################################################################

end
