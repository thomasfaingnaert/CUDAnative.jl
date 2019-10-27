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

        @testset "wmma_load_a" begin
            buf     = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float16))
            buf_ptr = convert(CuPtr{Float16}, buf)

            res = 42 * ones(Float16, 16 * 16)
            unsafe_copyto!(buf_ptr, pointer(res), 16 * 16)

            output = Mem.alloc(Mem.Device, sizeof(Bool))
            output_ptr = convert(CuPtr{Bool}, output)

            function kernel(buf_ptr, output_ptr)
                data = wmma_load_a(buf_ptr, 16)

                if all(val -> val == (VecElement{Float16}(42), VecElement{Float16}(42)), data)
                    unsafe_store!(output_ptr, 1)
                end

                return
            end

            @cuda threads=32 kernel(buf_ptr, output_ptr)

            result = zeros(Bool, 1)
            unsafe_copyto!(pointer(result), output_ptr, 1)

            @test all(result)
        end

        #= @testset "wmma_mma" begin =#
        #=     # Matmul kernel =#
        #=     function kernel(res, a, b) =#
        #=         a_frag = wmma_load_a(a, 16) =#
        #=         b_frag = wmma_load_b(b, 16) =#

        #=         res_frag = wmma_mma(a_frag.data0, a_frag.data1, a_frag.data2, a_frag.data3, a_frag.data4, a_frag.data5, a_frag.data6, a_frag.data7, b_frag.data0, b_frag.data1, b_frag.data2, b_frag.data3, b_frag.data4, b_frag.data5, b_frag.data6, b_frag.data7) =#

        #=         wmma_store_d(res, res_frag.data0, res_frag.data1, res_frag.data2, res_frag.data3, res_frag.data4, res_frag.data5, res_frag.data6, res_frag.data7, 16) =#
        #=         return =#
        #=     end =#

        #=     # Generate random input =#
        #=     a = rand(Float16, 16 * 16) =#
        #=     b = rand(Float16, 16 * 16) =#

        #=     # Allocate memory on GPU =#
        #=     dev_res     = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float32)) =#
        #=     dev_res_ptr = convert(CuPtr{Float32}, dev_res) =#

        #=     dev_a       = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float16)) =#
        #=     dev_a_ptr   = convert(CuPtr{Float16}, dev_a) =#

        #=     dev_b       = Mem.alloc(Mem.Device, 16 * 16 * sizeof(Float16)) =#
        #=     dev_b_ptr   = convert(CuPtr{Float16}, dev_b) =#

        #=     # Copy input to the GPU =#
        #=     unsafe_copyto!(dev_a_ptr, pointer(a), 16 * 16) =#
        #=     unsafe_copyto!(dev_b_ptr, pointer(b), 16 * 16) =#

        #=     # Perform multiplication =#
        #=     @cuda threads=32 kernel(dev_res_ptr, dev_a_ptr, dev_b_ptr) =#

        #=     # Check result =#
        #=     res = zeros(Float32, 16 * 16) =#
        #=     unsafe_copyto!(pointer(res), dev_res_ptr, 16 * 16) =#

        #=     for i=0:15 =#
        #=         for j=0:15 =#
        #=             tmp = 0 =#

        #=             for k=0:15 =#
        #=                 tmp += a[1 + 16 * i + k] * b[1 + 16 * k + j] =#
        #=             end =#

        #=             @test tmp ≈ res[1 + 16 * i + j] rtol=0.01 =#
        #=         end =#
        #=     end =#
        #= end =#
    end

################################################################################

end
