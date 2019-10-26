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

struct wmma_fragment
    data0::NTuple{2, VecElement{Float16}}
    data1::NTuple{2, VecElement{Float16}}
    data2::NTuple{2, VecElement{Float16}}
    data3::NTuple{2, VecElement{Float16}}
    data4::NTuple{2, VecElement{Float16}}
    data5::NTuple{2, VecElement{Float16}}
    data6::NTuple{2, VecElement{Float16}}
    data7::NTuple{2, VecElement{Float16}}
end

function wmma_load_a(src_addr, stride)
    #= ret = wmma_fragment((0, 0), (0, 0), (0, 0), (0, 0)) =#

    #= return ret =#

    return Base.llvmcall((
    """
    declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.load.a.sync.row.m16n16k16.stride.f16(i8*, i32)
    """,
    """
    %src_ptr = inttoptr i64 %0 to i8*

    %ret = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.load.a.sync.row.m16n16k16.stride.f16(i8* %src_ptr, i32 %1)

    %ret_0 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 0
    %ret_1 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 1
    %ret_2 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 2
    %ret_3 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 3
    %ret_4 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 4
    %ret_5 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 5
    %ret_6 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 6
    %ret_7 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 7

    %ret_0_conv = bitcast <2 x half> %ret_0 to <2 x i16>
    %ret_1_conv = bitcast <2 x half> %ret_1 to <2 x i16>
    %ret_2_conv = bitcast <2 x half> %ret_2 to <2 x i16>
    %ret_3_conv = bitcast <2 x half> %ret_3 to <2 x i16>
    %ret_4_conv = bitcast <2 x half> %ret_4 to <2 x i16>
    %ret_5_conv = bitcast <2 x half> %ret_5 to <2 x i16>
    %ret_6_conv = bitcast <2 x half> %ret_6 to <2 x i16>
    %ret_7_conv = bitcast <2 x half> %ret_7 to <2 x i16>

    %ret_aggr_0 = insertvalue { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } undef,       <2 x i16> %ret_0_conv, 0
    %ret_aggr_1 = insertvalue { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } %ret_aggr_0, <2 x i16> %ret_1_conv, 1
    %ret_aggr_2 = insertvalue { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } %ret_aggr_1, <2 x i16> %ret_2_conv, 2
    %ret_aggr_3 = insertvalue { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } %ret_aggr_2, <2 x i16> %ret_3_conv, 3
    %ret_aggr_4 = insertvalue { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } %ret_aggr_3, <2 x i16> %ret_4_conv, 4
    %ret_aggr_5 = insertvalue { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } %ret_aggr_4, <2 x i16> %ret_5_conv, 5
    %ret_aggr_6 = insertvalue { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } %ret_aggr_5, <2 x i16> %ret_6_conv, 6
    %ret_aggr_7 = insertvalue { <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16>, <2 x i16> } %ret_aggr_6, <2 x i16> %ret_7_conv, 7

    ret { <2 x i16 >, <2 x i16 >, <2 x i16 >, <2 x i16 >, <2 x i16 >, <2 x i16 >, <2 x i16 >, <2 x i16 > } %ret_aggr_7
    """),
    wmma_fragment,
    Tuple{Int64 ,Int32},
    convert(Int64, src_addr),
    convert(Int32, stride))
end
