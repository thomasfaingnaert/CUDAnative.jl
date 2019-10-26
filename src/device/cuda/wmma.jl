export wmma_store_d

wmma_store_d(dst_addr) =
    Base.llvmcall((
    """
    declare void @llvm.nvvm.wmma.store.d.sync.row.m16n16k16.stride.f32(i8*, float, float, float, float, float, float, float, float, i32)
    """,
    """
    %dst_ptr = inttoptr i64 %0 to i8*
    call void @llvm.nvvm.wmma.store.d.sync.row.m16n16k16.stride.f32(i8* %dst_ptr, float 42.0e+0, float 42.0e+0, float 42.0e+0, float 42.0e+0, float 42.0e+0, float 42.0e+0, float 42.0e+0, float 42.0e+0, i32 16)
    ret void
    """),
    Nothing, Tuple{Int64}, convert(Int64, dst_addr))
