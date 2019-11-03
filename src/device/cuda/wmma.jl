################################################################################

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

wmma_mma(a, b) =
    Base.llvmcall((
    "
    declare { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.mma.sync.col.col.m16n16k16.f32.f32(<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, float, float, float, float, float, float, float, float)
    ",
    "
    %a_00 = extractvalue [16 x i16] %0, 0
    %a_01 = extractvalue [16 x i16] %0, 1
    %a_02 = extractvalue [16 x i16] %0, 2
    %a_03 = extractvalue [16 x i16] %0, 3
    %a_04 = extractvalue [16 x i16] %0, 4
    %a_05 = extractvalue [16 x i16] %0, 5
    %a_06 = extractvalue [16 x i16] %0, 6
    %a_07 = extractvalue [16 x i16] %0, 7
    %a_08 = extractvalue [16 x i16] %0, 8
    %a_09 = extractvalue [16 x i16] %0, 9
    %a_10 = extractvalue [16 x i16] %0, 10
    %a_11 = extractvalue [16 x i16] %0, 11
    %a_12 = extractvalue [16 x i16] %0, 12
    %a_13 = extractvalue [16 x i16] %0, 13
    %a_14 = extractvalue [16 x i16] %0, 14
    %a_15 = extractvalue [16 x i16] %0, 15

    %b_00 = extractvalue [16 x i16] %1, 0
    %b_01 = extractvalue [16 x i16] %1, 1
    %b_02 = extractvalue [16 x i16] %1, 2
    %b_03 = extractvalue [16 x i16] %1, 3
    %b_04 = extractvalue [16 x i16] %1, 4
    %b_05 = extractvalue [16 x i16] %1, 5
    %b_06 = extractvalue [16 x i16] %1, 6
    %b_07 = extractvalue [16 x i16] %1, 7
    %b_08 = extractvalue [16 x i16] %1, 8
    %b_09 = extractvalue [16 x i16] %1, 9
    %b_10 = extractvalue [16 x i16] %1, 10
    %b_11 = extractvalue [16 x i16] %1, 11
    %b_12 = extractvalue [16 x i16] %1, 12
    %b_13 = extractvalue [16 x i16] %1, 13
    %b_14 = extractvalue [16 x i16] %1, 14
    %b_15 = extractvalue [16 x i16] %1, 15

    %a_00_conv = bitcast i16 %a_00 to half
    %a_01_conv = bitcast i16 %a_01 to half
    %a_02_conv = bitcast i16 %a_02 to half
    %a_03_conv = bitcast i16 %a_03 to half
    %a_04_conv = bitcast i16 %a_04 to half
    %a_05_conv = bitcast i16 %a_05 to half
    %a_06_conv = bitcast i16 %a_06 to half
    %a_07_conv = bitcast i16 %a_07 to half
    %a_08_conv = bitcast i16 %a_08 to half
    %a_09_conv = bitcast i16 %a_09 to half
    %a_10_conv = bitcast i16 %a_10 to half
    %a_11_conv = bitcast i16 %a_11 to half
    %a_12_conv = bitcast i16 %a_12 to half
    %a_13_conv = bitcast i16 %a_13 to half
    %a_14_conv = bitcast i16 %a_14 to half
    %a_15_conv = bitcast i16 %a_15 to half

    %b_00_conv = bitcast i16 %b_00 to half
    %b_01_conv = bitcast i16 %b_01 to half
    %b_02_conv = bitcast i16 %b_02 to half
    %b_03_conv = bitcast i16 %b_03 to half
    %b_04_conv = bitcast i16 %b_04 to half
    %b_05_conv = bitcast i16 %b_05 to half
    %b_06_conv = bitcast i16 %b_06 to half
    %b_07_conv = bitcast i16 %b_07 to half
    %b_08_conv = bitcast i16 %b_08 to half
    %b_09_conv = bitcast i16 %b_09 to half
    %b_10_conv = bitcast i16 %b_10 to half
    %b_11_conv = bitcast i16 %b_11 to half
    %b_12_conv = bitcast i16 %b_12 to half
    %b_13_conv = bitcast i16 %b_13 to half
    %b_14_conv = bitcast i16 %b_14 to half
    %b_15_conv = bitcast i16 %b_15 to half

    %a_vec_00 = insertelement <2 x half> undef,     half %a_00_conv, i64 0
    %a_vec_01 = insertelement <2 x half> %a_vec_00, half %a_01_conv, i64 1
    %a_vec_02 = insertelement <2 x half> undef,     half %a_02_conv, i64 0
    %a_vec_03 = insertelement <2 x half> %a_vec_02, half %a_03_conv, i64 1
    %a_vec_04 = insertelement <2 x half> undef,     half %a_04_conv, i64 0
    %a_vec_05 = insertelement <2 x half> %a_vec_04, half %a_05_conv, i64 1
    %a_vec_06 = insertelement <2 x half> undef,     half %a_06_conv, i64 0
    %a_vec_07 = insertelement <2 x half> %a_vec_06, half %a_07_conv, i64 1
    %a_vec_08 = insertelement <2 x half> undef,     half %a_08_conv, i64 0
    %a_vec_09 = insertelement <2 x half> %a_vec_08, half %a_09_conv, i64 1
    %a_vec_10 = insertelement <2 x half> undef,     half %a_10_conv, i64 0
    %a_vec_11 = insertelement <2 x half> %a_vec_10, half %a_11_conv, i64 1
    %a_vec_12 = insertelement <2 x half> undef,     half %a_12_conv, i64 0
    %a_vec_13 = insertelement <2 x half> %a_vec_12, half %a_13_conv, i64 1
    %a_vec_14 = insertelement <2 x half> undef,     half %a_14_conv, i64 0
    %a_vec_15 = insertelement <2 x half> %a_vec_14, half %a_15_conv, i64 1

    %b_vec_00 = insertelement <2 x half> undef,     half %b_00_conv, i64 0
    %b_vec_01 = insertelement <2 x half> %b_vec_00, half %b_01_conv, i64 1
    %b_vec_02 = insertelement <2 x half> undef,     half %b_02_conv, i64 0
    %b_vec_03 = insertelement <2 x half> %b_vec_02, half %b_03_conv, i64 1
    %b_vec_04 = insertelement <2 x half> undef,     half %b_04_conv, i64 0
    %b_vec_05 = insertelement <2 x half> %b_vec_04, half %b_05_conv, i64 1
    %b_vec_06 = insertelement <2 x half> undef,     half %b_06_conv, i64 0
    %b_vec_07 = insertelement <2 x half> %b_vec_06, half %b_07_conv, i64 1
    %b_vec_08 = insertelement <2 x half> undef,     half %b_08_conv, i64 0
    %b_vec_09 = insertelement <2 x half> %b_vec_08, half %b_09_conv, i64 1
    %b_vec_10 = insertelement <2 x half> undef,     half %b_10_conv, i64 0
    %b_vec_11 = insertelement <2 x half> %b_vec_10, half %b_11_conv, i64 1
    %b_vec_12 = insertelement <2 x half> undef,     half %b_12_conv, i64 0
    %b_vec_13 = insertelement <2 x half> %b_vec_12, half %b_13_conv, i64 1
    %b_vec_14 = insertelement <2 x half> undef,     half %b_14_conv, i64 0
    %b_vec_15 = insertelement <2 x half> %b_vec_14, half %b_15_conv, i64 1

    %res = call { float, float, float, float, float, float, float, float } @llvm.nvvm.wmma.mma.sync.col.col.m16n16k16.f32.f32( <2 x half> %a_vec_01, <2 x half> %a_vec_03, <2 x half> %a_vec_05, <2 x half> %a_vec_07, <2 x half> %a_vec_09, <2 x half> %a_vec_11, <2 x half> %a_vec_13, <2 x half> %a_vec_15, <2 x half> %b_vec_01, <2 x half> %b_vec_03, <2 x half> %b_vec_05, <2 x half> %b_vec_07, <2 x half> %b_vec_09, <2 x half> %b_vec_11, <2 x half> %b_vec_13, <2 x half> %b_vec_15, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0, float 0.0e+0)

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
    Tuple{NTuple{16, Float16}, NTuple{16, Float16}},
    convert(NTuple{16, Float16}, a),
    convert(NTuple{16, Float16}, b))

################################################################################
