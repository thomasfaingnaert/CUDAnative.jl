# NOTE: this is part of the @cuda test set, but needs to be top-level
module KernelModule
    export do_more_nothing
    @target ptx do_more_nothing() = return nothing
end

@testset "CUDA.jl native" begin

dev = CuDevice(0)
ctx = CuContext(dev)


@testset "code generation" begin
    @testset "LLVM IR" begin
        @target ptx foo() = return nothing
        ir = sprint(io->code_llvm(io, foo, (),
                                  #=strip_ir_metadata=#true, #=dump_module=#true))

        # module should contain only our function, returning void
        @test length(matchall(r"define .+", ir)) == 1
        @test ismatch(r"define void @julia_.+_foo_.+\(\) #0 \{", ir)
        # module should be created for the PTX back-end
        @test contains(ir, "!\"Julia Codegen Target\", !\"ptx\"")
        # function should be generated by the PTX back-end
        @test ismatch(r"attributes #0 = \{.+\"jl_cgtarget\"=\"ptx\".+\}", ir)
        # GC frame ref should have been optimized away
        @test !contains(ir, "jl_get_ptls_states")
    end

    @testset "PTX assembly" begin
        # TODO: PTX assembly generation / code_native
        # -> test if foo and bar doesn't end up in same PTX module

        # TODO: assert .entry
        # TODO: assert devfun non .entry
    end

    @testset "exceptions" begin
        @target ptx function throw_exception()
            throw(DivideError())
        end
        ir = sprint(io->code_llvm(io, throw_exception, ()))

        # exceptions should get lowered to a plain trap...
        @test contains(ir, "llvm.trap")
        # not a jl_throw referencing a jl_value_t representing the exception
        @test !contains(ir, "jl_value_t")
        @test !contains(ir, "jl_throw")
    end

    # delayed binding lookup (due to noexisting global)
    let
        @target ptx foo() = nonexisting
        @test_throws ErrorException code_native(DevNull, foo, ())
    end

    # generic call to nonexisting function
    let
        @target ptx foo() = nonexisting()
        @test_throws ErrorException code_native(DevNull, foo, ())
    end

    let
        @target ptx foo() = return nothing
        @test_throws ErrorException foo()
    end

    # bug: generate code twice for the same kernel (jl_to_ptx wasn't idempotent)
    let
        @target ptx foo() = return nothing
        code_native(DevNull, foo, ())
        code_native(DevNull, foo, ())
    end

    # TODO: we use a lot of foo in here, make sure if we redefine foo in an
    # inner clause it refers to the new function
end


@testset "@cuda" begin
    @target ptx do_nothing() = return nothing

    @test_throws UndefVarError @cuda (1, 1) undefined_kernel()

    @testset "kernel dims" begin
        @test_throws ArgumentError @cuda (0, 0) do_nothing()
        @cuda (1, 1) do_nothing()
    end

    @testset "external kernel" begin
        @cuda (1, 1) KernelModule.do_more_nothing()
        @eval begin
            using KernelModule
            @cuda (1, 1) do_more_nothing()
        end
    end

    @testset "return values" begin
        @target ptx retint() = return 1
        # FIXME: disabled because of #15043
        #@test_throws ErrorException @cuda (1, 1) retint()
    end

    @testset "argument passing" begin
        dims = (16, 16)
        len = prod(dims)

        @target ptx function array_copy(input::CuDeviceArray{Float32},
                                        output::CuDeviceArray{Float32})
            i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
            output[i] = input[i]

            return nothing
        end

        @testset "manual allocation" begin
            input = round(rand(Float32, dims) * 100)

            input_dev = CuArray(input)
            output_dev = CuArray(Float32, dims)

            @cuda (len, 1) array_copy(input_dev, output_dev)
            output = to_host(output_dev)
            @test_approx_eq input output

            free(input_dev)
            free(output_dev)
        end

        # Copy non-bit array
        @test_throws ArgumentError begin
            # Something that's certainly not a bit type
            f =  x -> x*x
            input = [f for i=1:10]
            cu_input = CuArray(input)
        end

        # CuArray with not-bit elements
        let
            @test_throws ArgumentError CuArray(Function, 10)
            @test_throws ArgumentError CuArray(Function, (10, 10))
        end

        # cu mem tests
        let
            @test_throws ArgumentError CUDA.cualloc(Function, 10)

            dev_array = CuArray(Int32, 10)
            CUDA.cumemset(dev_array.ptr, UInt32(0), 10)
            host_array = to_host(dev_array)

            for i in host_array
                @assert i == 0 "Memset failed on element $i"
            end

            CUDA.free(dev_array)

        end

        @testset "auto-managed host data" begin
            input = round(rand(Float32, dims) * 100)

            output1 = Array(Float32, dims)
            @cuda (len, 1) array_copy(CuIn(input), CuOut(output1))
            @test_approx_eq input output1

            output2 = Array(Float32, dims)
            # ... without specifying type
            @cuda (len, 1) array_copy(input, output2)
            @test_approx_eq input output2

            output3 = Array(Float32, dims)
            #  ... not using containers
            @cuda (len, 1) array_copy(round(input*100), output3)
            @test_approx_eq round(input*100) output3
        end

        @target ptx function array_lastvalue(a::CuDeviceArray{Float32},
                                             x::CuDeviceArray{Float32})
            i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
            max = gridDim().x * blockDim().x
            if i == max
                x[1] = a[i]
            end

            return nothing
        end

        # scalar through single-value array
        let
            arr = round(rand(Float32, dims) * 100)
            val = Float32[0]

            @cuda (len, 1) array_lastvalue(CuIn(arr), CuOut(val))
            @test_approx_eq arr[dims...] val[1]
        end

        @target ptx @noinline function array_lastvalue_devfun(a::CuDeviceArray{Float32},
                                                              x::CuDeviceArray{Float32})
            i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
            max = gridDim().x * blockDim().x
            if i == max
                x[1] = lastvalue_devfun(a, i)
            end

            return nothing
        end

        @target ptx function lastvalue_devfun(a::CuDeviceArray{Float32}, i)
            return a[i]
        end

        # same, but using a device function
        let
            arr = round(rand(Float32, dims) * 100)
            val = Float32[0]

            @cuda (len, 1) array_lastvalue_devfun(CuIn(arr), CuOut(val))
            @test_approx_eq arr[dims...] val[1]
        end
    end

end

@testset "get kernel function" begin

    @target ptx function vadd(a::CuDeviceArray{Int}, b::CuDeviceArray{Int}, c::CuDeviceArray{Int})
        i = (blockIdx().x -1) * blockDim().x + threadIdx().x
        c[i] = a[i] + b[i]
        return nothing
    end

    # Arguments
    n = 500
    d_a = CuArray(ones(Int, n))
    d_b = CuArray(ones(Int, n))
    d_c = CuArray(Int, n)

    # Get raw pointers
    d_a_ptr = d_a.ptr.inner
    d_b_ptr = d_b.ptr.inner
    d_c_ptr = d_b.ptr.inner

    # Get compiled kernel handle
    kernel = CUDA.get_kernel(vadd, d_a_ptr, d_b_ptr, d_c_ptr)

    #CUDA.exec((1, n, 0), kernel, d_a, d_b, d_c)
    CUDA.launch(kernel, 1, n, (d_a_ptr, d_b_ptr, d_c_ptr))

    c = to_host(d_c)
    result = fill(2::Int, n)
    @assert result == c

    free(d_a)
    free(d_b)
    free(d_c)
end

@testset "shared memory" begin
    dims = (16, 16)
    len = prod(dims)

    @target ptx function array_reverse(a::CuDeviceArray{Int64})
        chunk_size = Int32(blockDim().x)

        # Get shared memory
        tmp = cuSharedMem_i64()

        # Copy from a to shared mem
        i = (blockIdx().x -1) * chunk_size + threadIdx().x
        tmp_dst = chunk_size - threadIdx().x + 1
        setCuSharedMem_i64(tmp, tmp_dst, a[i])

        sync_threads()

        # Calculate destination, starting from 0
        dest_block = gridDim().x - blockIdx().x
        offset = dest_block * chunk_size
        dst_index = offset +  threadIdx().x

        a[dst_index] = getCuSharedMem_i64(tmp, threadIdx().x)

        return nothing
    end

    @target ptx function array_reverse(a::CuDeviceArray{Float32})
        chunk_size = Int32(blockDim().x)

        # Get shared memory
        tmp = cuSharedMem()

        # Copy from a to shared mem
        i = (blockIdx().x -1) * chunk_size + threadIdx().x
        tmp_dst = chunk_size - threadIdx().x + 1
        setCuSharedMem(tmp, tmp_dst, a[i])

        sync_threads()

        # Calculate destination, starting from 0
        dest_block = gridDim().x - blockIdx().x
        offset = dest_block * chunk_size
        dst_index = offset +  threadIdx().x

        a[dst_index] = getCuSharedMem(tmp, threadIdx().x)

        return nothing
    end

    @target ptx function array_reverse(a::CuDeviceArray{Float64})
        chunk_size = Int32(blockDim().x)

        # Get shared memory
        tmp = cuSharedMem_double()

        # Copy from a to shared mem
        i = (blockIdx().x -1) * chunk_size + threadIdx().x
        tmp_dst = chunk_size - threadIdx().x + 1
        setCuSharedMem_double(tmp, tmp_dst, a[i])

        sync_threads()

        # Calculate destination, starting from 0
        dest_block = gridDim().x - blockIdx().x
        offset = dest_block * chunk_size
        dst_index = offset +  threadIdx().x

        a[dst_index] = getCuSharedMem_double(tmp, threadIdx().x)

        return nothing
    end

    # Params
    grid_size = 4
    block_size = 512
    n = grid_size * block_size

    # Test for multiple types
    types = [Int64, Float32, Float64]

    for T in types
        # Create data
        a = rand(T, n)
        r = reverse(a)  # to check later

        # call kernel
        shared_bytes = block_size * sizeof(T)
        @cuda (grid_size, block_size, shared_bytes) array_reverse(CuInOut(a))

        @assert a == r
    end
end


destroy(ctx)

end
