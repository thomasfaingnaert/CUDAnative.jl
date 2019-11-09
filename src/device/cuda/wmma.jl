export wmma_store_d, wmma_mma

################################################################################
# CONSTANTS
################################################################################

# Maps PTX types to LLVM types
map_ptx_to_llvm = Dict(
                       "f16" => "<2 x half>",
                       "f32" => "float"
                      )

# Maps PTX types to the LLVM type that llvmcall expects
map_ptx_to_llvmcall = Dict(
                       "f16" => "<2 x i16>",
                       "f32" => "float"
                      )

# Maps PTX types to Julia types
map_ptx_to_jl = Dict(
                     "f16" => NTuple{2, VecElement{Float16}},
                     "f32" => Float32
                    )

# Maps matrix & PTX types to fragment sizes
map_frag_sizes = Dict(
                      "a.f16" => 8,
                      "b.f16" => 8,
                      "c.f16" => 4,
                      "c.f32" => 8
                     )

################################################################################
# HELPER FUNCTIONS
################################################################################

macro gen_ir(template, count, delim="\n")
    return quote
        join([$(esc(template)) for $(esc(:i)) in 0:$(esc(count))-1], $(esc(delim)))
    end
end

function join_nonempty(args...)
    delim = args[end]
    arr = [args[1:end-1]...]

    return join(arr[arr .!= ""], delim)
end

get_llvm_ty(matrix, ptx_el_type) = map_ptx_to_llvm[ptx_el_type]

function get_struct_ty(matrix, ptx_el_type)
    llvm_ty = get_llvm_ty(matrix, ptx_el_type)
    frag_size = get_frag_sz(matrix, ptx_el_type)

    return "{ $(join(fill(llvm_ty, frag_size), ", ")) }"
end

get_llvmcall_ty(matrix, ptx_el_type) = map_ptx_to_llvmcall[ptx_el_type]

get_jl_ty(matrix, ptx_el_type) = map_ptx_to_jl[ptx_el_type]

get_frag_sz(matrix, ptx_el_type) = map_frag_sizes["$matrix.$ptx_el_type"]

################################################################################
# MATRIX LOAD
################################################################################

for mat in ["a", "b", "c"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]

    # TODO: Non-stride versions?

    # Float32 is only supported for C
    if (elem_type == "f32") && (mat != "c")
        continue
    end

    # Name of the Julia wrapper function
    func_name = Symbol(join_nonempty("llvm", "wmma", "load", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Name of the LLVM intrinsic
    llvm_intr = join_nonempty("@llvm", "nvvm", "wmma", "load", mat, "sync", layout, shape, addr_space, stride, elem_type, ".")

    # Determine types for this (matrix, elem_type) combination
    sz = get_frag_sz(mat, elem_type)
    llvm_ty = get_llvm_ty(mat, elem_type)
    struct_ty = get_struct_ty(mat, elem_type)
    lc_ty = get_llvmcall_ty(mat, elem_type)
    jl_ty = get_jl_ty(mat, elem_type)

    # Generate LLVM IR
    ir = ("declare $struct_ty $llvm_intr(i8*, i32)",
    "
    %src_ptr = inttoptr i64 %0 to i8*

    %ret.llvm = call $struct_ty $llvm_intr(i8* %src_ptr, i32 %1)

    $(@gen_ir("%ret.llvm.$i = extractvalue $struct_ty %ret.llvm, $i", sz))

    $(@gen_ir("%ret.jl.$i= bitcast $llvm_ty %ret.llvm.$i to $lc_ty", sz))

    $(@gen_ir("%ret.aggr.$i = insertvalue [$sz x $lc_ty] $(i == 0 ? "undef" : "%ret.aggr.$(i-1)"), $lc_ty %ret.jl.$i, $i", sz))

    ret [$sz x $lc_ty] %ret.aggr.$(sz-1)
    ")

    @eval $func_name(src_addr, stride) = Base.llvmcall($ir,
        NTuple{$sz, $jl_ty},
        Tuple{Int64, Int32},
        convert(Int64, src_addr),
        convert(Int32, stride))

    @eval export $func_name
end

################################################################################
# MATRIX STORE
################################################################################

wmma_store_d(dst_addr, data_0, data_1, data_2, data_3, data_4, data_5, data_6, data_7, stride) =
    Base.llvmcall((
    "
    declare void @llvm.nvvm.wmma.store.d.sync.col.m16n16k16.stride.f32(i8*, float, float, float, float, float, float, float, float, i32)
    ",
    "
    %dst_ptr = inttoptr i64 %0 to i8*
    call void @llvm.nvvm.wmma.store.d.sync.col.m16n16k16.stride.f32(i8* %dst_ptr, float %1, float %2, float %3, float %4, float %5, float %6, float %7, float %8, i32 %9)
    ret void
    "),
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

################################################################################
# MATRIX MULTIPLY ACCUMULATE
################################################################################

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
