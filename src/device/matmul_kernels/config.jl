struct Config{
    #= Params =#
    MATMUL_SHAPE,                 # MNK, overall shape of the MATMUL operation
    BLOCK_SHAPE,                # MNK, shape of each CTA tile
    WARPS_PER_BLOCK,            # scalar, number of warps per CTA

    MEM_A_WARP,                 # MK, shape of each warp tile during memory operations involving matrix A
    MEM_A_THREAD,               # MK, shape of each thread tile during memory operations involving matrix A

    MEM_B_WARP,                 # KN, shape of each warp tile during memory operations involving matrix B
    MEM_B_THREAD,               # KN, shape of each thread tile during memory operations involving matrix B

    MEM_CD_WARP,                # MN, shape of each warp tile during memory operations involving matrix C or D
    MEM_CD_THREAD,              # MN, shape of each thread tile during memory operations involving matrix C or D

    COMPUTE_WARP,               # MNK, shape of each warp tile during the inner loop computations
    COMPUTE_OP_SHAPE,           # MNK, shape of the operation used in the inner loop

    #= Layouts =#
    GLOBAL_A_LAYOUT,            # layout of the A matrix in global memory
    GLOBAL_B_LAYOUT,            # layout of the B matrix in global memory
    GLOBAL_C_LAYOUT,            # layout of the C matrix in global memory
    GLOBAL_D_LAYOUT,            # layout of the D matrix in global memory

    SHARED_A_LAYOUT,            # layout of the A matrix in shared memory
    SHARED_B_LAYOUT,            # layout of the B matrix in shared memory
    SHARED_C_LAYOUT,            # layout of the C matrix in shared memory
    SHARED_D_LAYOUT,            # layout of the D matrix in shared memory

    #= Operator =#
    OPERATOR,                   # which operator to use in the inner loop
   }
end

@inline function Base.getproperty(conf::Type{Config{MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR}}, sym::Symbol) where {MATMUL_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR}
    if sym == :launch_args
        return (threads = WARPS_PER_BLOCK * 32, blocks = (MATMUL_SHAPE.M ÷ BLOCK_SHAPE.M, MATMUL_SHAPE.N ÷ BLOCK_SHAPE.N), shmem = 64 * 1024)
    else
        return getfield(conf, sym)
    end
end

function get_config(; gemm_shape, kwargs...)
    params = Dict(kwargs)

    return Config{
        #= Params =#
        gemm_shape,
        get(params, :block_shape, (M = 128, N = 128, K = 64)),
        get(params, :warps_per_block, 8),
        get(params, :mem_a_warp, (M = 128, K = 2)),
        get(params, :mem_a_thread, (M = 8, K = 1)),
        get(params, :mem_b_warp, (K = 64, N = 4)),
        get(params, :mem_b_thread, (K = 8, N = 1)),
        get(params, :mem_cd_warp, (M = 128, N = 1)),
        get(params, :mem_cd_thread, (M = 4, N = 1)),
        get(params, :compute_warp, (M = 32, N = 64, K = 16)),
        op_shape(get(params, :operator, WMMAOp{16, 16, 16})),

        #= Layouts =#
        get(params, :global_a_layout, AlignedColMajor{Float16}),
        get(params, :global_b_layout, AlignedColMajor{Float16}),
        get(params, :global_c_layout, AlignedColMajor{Float32}),
        get(params, :global_d_layout, AlignedColMajor{Float32}),

        get(params, :shared_a_layout, Padded{AlignedColMajor{Float16}, 8}),
        get(params, :shared_b_layout, Padded{AlignedColMajor{Float16}, 8}),
        get(params, :shared_c_layout, AlignedColMajor{Float32}),
        get(params, :shared_d_layout, AlignedColMajor{Float32}),

        #= Operators =#
        get(params, :operator, WMMAOp{16, 16, 16}),
    }
end
