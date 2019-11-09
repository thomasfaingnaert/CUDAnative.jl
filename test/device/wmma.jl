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

                # Type-dependent variables
                array_ty = elem_type == "f16" ? Float16 : Float32
                expected = elem_type == "f16" ? (VecElement{Float16}(42), VecElement{Float16}(42)) : Float32(42)

                # Get the function name
                func = getfield(Main, Symbol("llvm_wmma_load_$(mat)_$(layout)_$(shape)_stride_$(elem_type)"))

                input      = 42 * ones(array_ty, (16, 16))
                input_dev  = CuArray(input)
                result     = Array{Bool}(undef, 1)
                result_dev = CuArray(result)

                function kernel(input_dev, result_dev)
                    data = func(pointer(input_dev), 16)
                    result_dev[1] = all(val -> val == expected, data)
                    return
                end

                @cuda threads=32 kernel(input_dev, result_dev)
                @test all(Array(result_dev))
            end
        end

        @testset "llvm_wmma_store" begin
            @testset "$(mat)_$(layout)_$(shape)_$(addr_space)_$(elem_type)" for mat in ["d"],
                layout in ["row", "col"],
                shape in ["m16n16k16"],
                addr_space in [""],
                stride in ["stride"],
                elem_type in ["f16", "f32"]

                # TODO: Test address space?

                # Type-dependent variables
                array_ty = elem_type == "f16" ? Float16 : Float32
                data = elem_type == "f16" ?
                    (
                       (VecElement{Float16}(42), VecElement{Float16}(42)),
                       (VecElement{Float16}(42), VecElement{Float16}(42)),
                       (VecElement{Float16}(42), VecElement{Float16}(42)),
                       (VecElement{Float16}(42), VecElement{Float16}(42))
                    ) : (42, 42, 42, 42, 42, 42, 42, 42)

                # Get the function name
                func = getfield(Main, Symbol("llvm_wmma_store_$(mat)_$(layout)_$(shape)_stride_$(elem_type)"))

                output     = Array{array_ty}(undef, (16, 16))
                output_dev = CuArray(output)

                function kernel(output_dev)
                    func(pointer(output_dev), data, 16)
                    return
                end

                @cuda threads=32 kernel(output_dev)
                @test all(Array(output_dev) .== 42.0)
            end
        end

        @testset "wmma_mma" begin
            # Generate input matrices
            a     = rand(Float16, (16, 16))
            a_dev = CuArray(a)
            b     = rand(Float16, (16, 16))
            b_dev = CuArray(b)
            c     = rand(Float32, (16, 16))
            c_dev = CuArray(c)

            # Reserve space for result
            d     = Array{Float32}(undef, (16, 16))
            d_dev = CuArray(d)

            # Matrix MAC kernel (D = A * B + C)
            function kernel(a_dev, b_dev, c_dev, d_dev)
                a_frag = llvm_wmma_load_a_col_m16n16k16_stride_f16(pointer(a_dev), 16)
                b_frag = llvm_wmma_load_b_col_m16n16k16_stride_f16(pointer(b_dev), 16)
                c_frag = llvm_wmma_load_c_col_m16n16k16_stride_f32(pointer(c_dev), 16)

                d_frag = llvm_wmma_mma_col_col_m16n16k16_f32_f32(a_frag, b_frag, c_frag)

                llvm_wmma_store_d_col_m16n16k16_stride_f32(pointer(d_dev), d_frag, 16)
                return
            end

            @cuda threads=32 kernel(a_dev, b_dev, c_dev, d_dev)
            @test a * b + c ≈ Array(d_dev) rtol=0.01
        end
    end

################################################################################

end
