export wmma_store_d, wmma_load_a, wmma_load_b, wmma_mma

wmma_store_d(dst_addr, data, stride) =
    Base.llvmcall((
    "
    declare void @llvm.nvvm.wmma.store.d.sync.col.m16n16k16.stride.f32(i8*, float, float, float, float, float, float, float, float, i32)
    ",
    "
    %dst_ptr = inttoptr i64 %0 to i8*

    %data_0 = extractvalue [8 x float] %1, 0
    %data_1 = extractvalue [8 x float] %1, 1
    %data_2 = extractvalue [8 x float] %1, 2
    %data_3 = extractvalue [8 x float] %1, 3
    %data_4 = extractvalue [8 x float] %1, 4
    %data_5 = extractvalue [8 x float] %1, 5
    %data_6 = extractvalue [8 x float] %1, 6
    %data_7 = extractvalue [8 x float] %1, 7

    call void @llvm.nvvm.wmma.store.d.sync.col.m16n16k16.stride.f32(i8* %dst_ptr, float %data_0, float %data_1, float %data_2, float %data_3, float %data_4, float %data_5, float %data_6, float %data_7, i32 %2)

    ret void
    "),
    Nothing,
    Tuple{Int64, NTuple{8, Float32}, Int32},
    convert(Int64, dst_addr),
    convert(NTuple{8, Float32}, data),
    convert(Int32, stride))

for matrix in (:a, :b)
    func_name = Symbol("wmma_load_", matrix)
    matrix_str = string(matrix)

    ir = ("declare { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.load.$matrix_str.sync.col.m16n16k16.stride.f16(i8*, i32)",
    "
    %src_ptr = inttoptr i64 %0 to i8*

    %ret = call { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } @llvm.nvvm.wmma.load.$matrix_str.sync.col.m16n16k16.stride.f16(i8* %src_ptr, i32 %1)

    %ret_0 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 0
    %ret_1 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 1
    %ret_2 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 2
    %ret_3 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 3
    %ret_4 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 4
    %ret_5 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 5
    %ret_6 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 6
    %ret_7 = extractvalue { <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half> } %ret, 7

    %ret_flattened_00 = extractelement <2 x half> %ret_0, i32 0
    %ret_flattened_01 = extractelement <2 x half> %ret_0, i32 1
    %ret_flattened_02 = extractelement <2 x half> %ret_1, i32 0
    %ret_flattened_03 = extractelement <2 x half> %ret_1, i32 1
    %ret_flattened_04 = extractelement <2 x half> %ret_2, i32 0
    %ret_flattened_05 = extractelement <2 x half> %ret_2, i32 1
    %ret_flattened_06 = extractelement <2 x half> %ret_3, i32 0
    %ret_flattened_07 = extractelement <2 x half> %ret_3, i32 1
    %ret_flattened_08 = extractelement <2 x half> %ret_4, i32 0
    %ret_flattened_09 = extractelement <2 x half> %ret_4, i32 1
    %ret_flattened_10 = extractelement <2 x half> %ret_5, i32 0
    %ret_flattened_11 = extractelement <2 x half> %ret_5, i32 1
    %ret_flattened_12 = extractelement <2 x half> %ret_6, i32 0
    %ret_flattened_13 = extractelement <2 x half> %ret_6, i32 1
    %ret_flattened_14 = extractelement <2 x half> %ret_7, i32 0
    %ret_flattened_15 = extractelement <2 x half> %ret_7, i32 1

    %ret_flattened_00_conv = bitcast half %ret_flattened_00 to i16
    %ret_flattened_01_conv = bitcast half %ret_flattened_01 to i16
    %ret_flattened_02_conv = bitcast half %ret_flattened_02 to i16
    %ret_flattened_03_conv = bitcast half %ret_flattened_03 to i16
    %ret_flattened_04_conv = bitcast half %ret_flattened_04 to i16
    %ret_flattened_05_conv = bitcast half %ret_flattened_05 to i16
    %ret_flattened_06_conv = bitcast half %ret_flattened_06 to i16
    %ret_flattened_07_conv = bitcast half %ret_flattened_07 to i16
    %ret_flattened_08_conv = bitcast half %ret_flattened_08 to i16
    %ret_flattened_09_conv = bitcast half %ret_flattened_09 to i16
    %ret_flattened_10_conv = bitcast half %ret_flattened_10 to i16
    %ret_flattened_11_conv = bitcast half %ret_flattened_11 to i16
    %ret_flattened_12_conv = bitcast half %ret_flattened_12 to i16
    %ret_flattened_13_conv = bitcast half %ret_flattened_13 to i16
    %ret_flattened_14_conv = bitcast half %ret_flattened_14 to i16
    %ret_flattened_15_conv = bitcast half %ret_flattened_15 to i16

    %aggr_00 = insertvalue [16 x i16] undef,    i16 %ret_flattened_00_conv, 0
    %aggr_01 = insertvalue [16 x i16] %aggr_00, i16 %ret_flattened_01_conv, 1
    %aggr_02 = insertvalue [16 x i16] %aggr_01, i16 %ret_flattened_02_conv, 2
    %aggr_03 = insertvalue [16 x i16] %aggr_02, i16 %ret_flattened_03_conv, 3
    %aggr_04 = insertvalue [16 x i16] %aggr_03, i16 %ret_flattened_04_conv, 4
    %aggr_05 = insertvalue [16 x i16] %aggr_04, i16 %ret_flattened_05_conv, 5
    %aggr_06 = insertvalue [16 x i16] %aggr_05, i16 %ret_flattened_06_conv, 6
    %aggr_07 = insertvalue [16 x i16] %aggr_06, i16 %ret_flattened_07_conv, 7
    %aggr_08 = insertvalue [16 x i16] %aggr_07, i16 %ret_flattened_08_conv, 8
    %aggr_09 = insertvalue [16 x i16] %aggr_08, i16 %ret_flattened_09_conv, 9
    %aggr_10 = insertvalue [16 x i16] %aggr_09, i16 %ret_flattened_10_conv, 10
    %aggr_11 = insertvalue [16 x i16] %aggr_10, i16 %ret_flattened_11_conv, 11
    %aggr_12 = insertvalue [16 x i16] %aggr_11, i16 %ret_flattened_12_conv, 12
    %aggr_13 = insertvalue [16 x i16] %aggr_12, i16 %ret_flattened_13_conv, 13
    %aggr_14 = insertvalue [16 x i16] %aggr_13, i16 %ret_flattened_14_conv, 14
    %aggr_15 = insertvalue [16 x i16] %aggr_14, i16 %ret_flattened_15_conv, 15

    ret [16 x i16] %aggr_15
    ")

    @eval $func_name(src_addr, stride) = Base.llvmcall($ir,
        NTuple{16, Float16},
        Tuple{Int64 ,Int32},
        convert(Int64, src_addr),
        convert(Int32, stride))
end

wmma_mma(a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7) =
    Base.llvmcall((
    "
    declare { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.mma.sync.col.col.m16n16k16.f32.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float, float, float, float, float)
    ",
    "
    %conv_0 = bitcast <2 x i16> %0 to <2 x half>
    %conv_1 = bitcast <2 x i16> %1 to <2 x half>
    %conv_2 = bitcast <2 x i16> %2 to <2 x half>
    %conv_3 = bitcast <2 x i16> %3 to <2 x half>
    %conv_4 = bitcast <2 x i16> %4 to <2 x half>
    %conv_5 = bitcast <2 x i16> %5 to <2 x half>
    %conv_6 = bitcast <2 x i16> %6 to <2 x half>
    %conv_7 = bitcast <2 x i16> %7 to <2 x half>
    %conv_8 = bitcast <2 x i16> %8 to <2 x half>
    %conv_9 = bitcast <2 x i16> %9 to <2 x half>
    %conv_10 = bitcast <2 x i16> %10 to <2 x half>
    %conv_11 = bitcast <2 x i16> %11 to <2 x half>
    %conv_12 = bitcast <2 x i16> %12 to <2 x half>
    %conv_13 = bitcast <2 x i16> %13 to <2 x half>
    %conv_14 = bitcast <2 x i16> %14 to <2 x half>
    %conv_15 = bitcast <2 x i16> %15 to <2 x half>

    %res = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.mma.sync.col.col.m16n16k16.f32.f32( <2 x half> %conv_0, <2 x half> %conv_1, <2 x half> %conv_2, <2 x half> %conv_3, <2 x half> %conv_4, <2 x half> %conv_5, <2 x half> %conv_6, <2 x half> %conv_7, <2 x half> %conv_8, <2 x half> %conv_9, <2 x half> %conv_10, <2 x half> %conv_11, <2 x half> %conv_12, <2 x half> %conv_13, <2 x half> %conv_14, <2 x half> %conv_15, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0)

    %res_0 = extractvalue { float, float, float, float, float, float, float, float } %res, 0
    %res_1 = extractvalue { float, float, float, float, float, float, float, float } %res, 1
    %res_2 = extractvalue { float, float, float, float, float, float, float, float } %res, 2
    %res_3 = extractvalue { float, float, float, float, float, float, float, float } %res, 3
    %res_4 = extractvalue { float, float, float, float, float, float, float, float } %res, 4
    %res_5 = extractvalue { float, float, float, float, float, float, float, float } %res, 5
    %res_6 = extractvalue { float, float, float, float, float, float, float, float } %res, 6
    %res_7 = extractvalue { float, float, float, float, float, float, float, float } %res, 7

    %ret_0 = insertvalue [8 x float] undef,  float %res_0, 0
    %ret_1 = insertvalue [8 x float] %ret_0, float %res_1, 1
    %ret_2 = insertvalue [8 x float] %ret_1, float %res_2, 2
    %ret_3 = insertvalue [8 x float] %ret_2, float %res_3, 3
    %ret_4 = insertvalue [8 x float] %ret_3, float %res_4, 4
    %ret_5 = insertvalue [8 x float] %ret_4, float %res_5, 5
    %ret_6 = insertvalue [8 x float] %ret_5, float %res_6, 6
    %ret_7 = insertvalue [8 x float] %ret_6, float %res_7, 7

    ret [8 x float] %ret_7
    "),
    NTuple{8, Float32},
    Tuple{NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}, NTuple{2, VecElement{Float16}}},
    convert(NTuple{2, VecElement{Float16}}, a_0),
    convert(NTuple{2, VecElement{Float16}}, a_1),
    convert(NTuple{2, VecElement{Float16}}, a_2),
    convert(NTuple{2, VecElement{Float16}}, a_3),
    convert(NTuple{2, VecElement{Float16}}, a_4),
    convert(NTuple{2, VecElement{Float16}}, a_5),
    convert(NTuple{2, VecElement{Float16}}, a_6),
    convert(NTuple{2, VecElement{Float16}}, a_7),
    convert(NTuple{2, VecElement{Float16}}, b_0),
    convert(NTuple{2, VecElement{Float16}}, b_1),
    convert(NTuple{2, VecElement{Float16}}, b_2),
    convert(NTuple{2, VecElement{Float16}}, b_3),
    convert(NTuple{2, VecElement{Float16}}, b_4),
    convert(NTuple{2, VecElement{Float16}}, b_5),
    convert(NTuple{2, VecElement{Float16}}, b_6),
    convert(NTuple{2, VecElement{Float16}}, b_7))
