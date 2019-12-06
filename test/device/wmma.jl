expected = ntuple(i -> VecElement{Float16}(42), 8)

# Generate input matrices
a     = rand(Float16, (16, 16))
a_dev = CuArray(a)

# Matrix MAC kernel (D = A * B + C)
function kernel(a_dev)
    a_frag = llvm_wmma_load_a_row_m16n16k16_stride_f16(pointer(a_dev), 16)
    return
end

@cuda threads=32 kernel(a_dev)

CUDAnative.unflatten(NTuple{8, VecElement{Float16}}, ntuple(i -> Float16(i), 8))
