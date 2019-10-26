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

    end

################################################################################

end
