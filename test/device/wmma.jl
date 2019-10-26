@testset "WMMA" begin

################################################################################

    @testset "LLVM intrinsics" begin

        @testset "wmma_store_d" begin
            buf     = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float32))
            buf_ptr = convert(Int64, convert(CUDAdrv.CuPtr{Float32}, buf))

            function kernel(buf_ptr)
                CUDAnative.wmma_store_d(buf_ptr)
                return
            end

            @cuda threads=32 kernel(buf_ptr)

            res = zeros(Float32, 16 * 16)
            unsafe_copyto!(pointer(res), convert(CuPtr{Float32}, buf), 16 * 16)

            @test all(res .== 42.0)
        end

    end

################################################################################

end
