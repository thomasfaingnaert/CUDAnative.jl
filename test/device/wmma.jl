@testset "WMMA" begin

################################################################################

    @testset "LLVM intrinsics" begin

        @testset "llvm_wmma_load" begin

            @testset "$(mat)_$(layout)_$(shape)_$(addr_space)_$(elem_type)" for mat in ["a", "b", "c"],
                layout in ["row", "col"],
                shape in ["m16n16k16"],
                addr_space in [""],
                stride in ["stride"],
                elem_type in ["f16", "f32"]

                # TODO: Test address space?

                # Float32 is only supported for C
                if (elem_type == "f32") && (mat != "c")
                    continue
                end

                # Get the function name
                func = getfield(Main, Symbol("llvm_wmma_load_$(mat)_$(layout)_$(shape)_stride_f16"))

                input      = 42 * ones(Float16, (16, 16))
                input_dev  = CuArray(input)
                result     = Array{Bool}(undef, 1)
                result_dev = CuArray(result)

                function kernel(input_dev, result_dev)
                    data = func(pointer(input_dev), 16)

                    result_dev[1] = all(val -> val == (VecElement{Float16}(42), VecElement{Float16}(42)), data)

                    return
                end

                @cuda threads=32 kernel(input_dev, result_dev)

                @test all(Array(result_dev))
            end
        end

        @testset "wmma_store_d" begin
            output     = Array{Float32}(undef, (16, 16))
            output_dev = CuArray(output)

            function kernel(output_dev)
                data = (42, 42, 42, 42, 42, 42, 42, 42)
                wmma_store_d(pointer(output_dev), data..., 16)
                return
            end

            @cuda threads=32 kernel(output_dev)

            @test all(Array(output_dev) .== 42.0)
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
                a_frag = llvm_wmma_load_a_col_m16n16k16_stride_f16(pointer(a_dev), 16)
                b_frag = llvm_wmma_load_b_col_m16n16k16_stride_f16(pointer(b_dev), 16)
                d_frag = wmma_mma(a_frag..., b_frag...)
                wmma_store_d(pointer(d_dev), d_frag..., 16)

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

end
