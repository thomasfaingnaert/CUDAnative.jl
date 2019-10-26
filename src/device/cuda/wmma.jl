export wmma_store_d

wmma_store_d(dst_addr, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, stride) =
    Base.llvmcall((
    """
    declare void @llvm.nvvm.wmma.store.d.sync.row.m16n16k16.stride.f32(i8*, float, float, float, float, float, float, float, float, i32)
    """,
    """
    %dst_ptr = inttoptr i64 %0 to i8*
    call void @llvm.nvvm.wmma.store.d.sync.row.m16n16k16.stride.f32(i8* %dst_ptr, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8, i32 %9)
    ret void
    """),
    Nothing,
    Tuple{Int64, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32, Int32},
    convert(Int64, dst_addr),
    convert(Float32, data_0),
    convert(Float32, data_1),
    convert(Float32, data_2),
    convert(Float32, data_3),
    convert(Float32, data_4),
    convert(Float32, data_5),
    convert(Float32, data_6),
    convert(Float32, data_7),
    convert(Int32, stride))
