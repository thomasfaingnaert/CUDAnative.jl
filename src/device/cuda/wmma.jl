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
                      "c.f32" => 8,
                      "d.f16" => 4,
                      "d.f32" => 8
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

get_llvmcall_ty(matrix, ptx_el_type) = map_ptx_to_llvmcall[ptx_el_type]

get_jl_ty(matrix, ptx_el_type) = map_ptx_to_jl[ptx_el_type]

get_frag_sz(matrix, ptx_el_type) = map_frag_sizes["$matrix.$ptx_el_type"]

################################################################################
# LOW LEVEL API
################################################################################

# -----------
# Matrix load
# -----------

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
    struct_ty = "{ $(@gen_ir(llvm_ty, sz, ", ")) }"
    lc_ty = get_llvmcall_ty(mat, elem_type)
    jl_ty = get_jl_ty(mat, elem_type)

    # Generate LLVM IR
    ir = ("declare $struct_ty $llvm_intr(i8*, i32)",
    "
    %src_ptr = inttoptr i64 %0 to i8*

    %ret.llvm = call $struct_ty $llvm_intr(i8* %src_ptr, i32 %1)

    $(@gen_ir("%ret.llvm.$i = extractvalue $struct_ty %ret.llvm, $i", sz))

    $(@gen_ir("%ret.jl.$i = bitcast $llvm_ty %ret.llvm.$i to $lc_ty", sz))

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

# ------------
# Matrix store
# ------------

for mat in ["d"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]

    # TODO: Non-stride versions?

    # Name of the Julia wrapper function
    func_name = Symbol(join_nonempty("llvm", "wmma", "store", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Name of the LLVM intrinsic
    llvm_intr = join_nonempty("@llvm", "nvvm", "wmma", "store", mat, "sync", layout, shape, addr_space, stride, elem_type, ".")

    # Determine types for this (matrix, elem_type) combination
    sz = get_frag_sz(mat, elem_type)
    llvm_ty = get_llvm_ty(mat, elem_type)
    lc_ty = get_llvmcall_ty(mat, elem_type)
    jl_ty = get_jl_ty(mat, elem_type)

    # Generate LLVM IR
    ir = ("declare void $llvm_intr(i8*, $(@gen_ir("$llvm_ty", sz, ", ")), i32)",
    "
    %dst_ptr = inttoptr i64 %0 to i8*

    $(@gen_ir("%data.jl.$i = extractvalue [$sz x $lc_ty] %1, $i", sz))

    $(@gen_ir("%data.llvm.$i = bitcast $lc_ty %data.jl.$i to $llvm_ty", sz))

    call void $llvm_intr(i8* %dst_ptr, $(@gen_ir("$llvm_ty %data.llvm.$i", sz, ", ")) , i32 %2)
    ret void
    ")

    @eval $func_name(dst_addr, data, stride) = Base.llvmcall($ir,
        Nothing,
        Tuple{Int64, NTuple{$sz, $jl_ty}, Int32},
        convert(Int64, dst_addr),
        convert(NTuple{$sz, $jl_ty}, data),
        convert(Int32, stride))

    @eval export $func_name
end

# --------------------------
# Matrix multiply accumulate
# --------------------------

for a_layout in ["col", "row"],
    b_layout in ["col", "row"],
    shape in ["m16n16k16"],
    d_elem_type in ["f16", "f32"],
    c_elem_type in ["f16", "f32"],
    b_elem_type in ["f16"],
    a_elem_type in ["f16"]

    # Name of the Julia wrapper function
    func_name = Symbol(join_nonempty("llvm", "wmma", "mma", a_layout, b_layout, shape, d_elem_type, c_elem_type, "_"))

    # Name of the LLVM intrinsic
    llvm_intr = join_nonempty("@llvm", "nvvm", "wmma", "mma", "sync", a_layout, b_layout, shape, d_elem_type, c_elem_type, ".")

    # Determine types for the (matrix, elem_type) combinations for matrix A
    a_sz = get_frag_sz("a", a_elem_type)
    a_llvm_ty = get_llvm_ty("a", a_elem_type)
    a_lc_ty = get_llvmcall_ty("a", a_elem_type)
    a_jl_ty = get_jl_ty("a", a_elem_type)

    # Determine types for the (matrix, elem_type) combinations for matrix B
    b_sz = get_frag_sz("b", b_elem_type)
    b_llvm_ty = get_llvm_ty("b", b_elem_type)
    b_lc_ty = get_llvmcall_ty("b", b_elem_type)
    b_jl_ty = get_jl_ty("b", b_elem_type)

    # Determine types for the (matrix, elem_type) combinations for matrix C
    c_sz = get_frag_sz("c", c_elem_type)
    c_llvm_ty = get_llvm_ty("c", c_elem_type)
    c_lc_ty = get_llvmcall_ty("c", c_elem_type)
    c_jl_ty = get_jl_ty("c", c_elem_type)

    # Determine types for the (matrix, elem_type) combinations for matrix D
    d_sz = get_frag_sz("d", d_elem_type)
    d_llvm_ty = get_llvm_ty("d", d_elem_type)
    d_lc_ty = get_llvmcall_ty("d", d_elem_type)
    d_jl_ty = get_jl_ty("d", d_elem_type)
    d_struct_ty = "{ $(@gen_ir(d_llvm_ty, d_sz, ", ")) }"

    # Create the argument string to the IR call
    args = join([
                @gen_ir("$a_llvm_ty %a.llvm.$i", a_sz, ", "),
                @gen_ir("$b_llvm_ty %b.llvm.$i", b_sz, ", "),
                @gen_ir("$c_llvm_ty %c.llvm.$i", c_sz, ", ")]
                , ", ")

    # Generate LLVM IR
    ir = ("declare $d_struct_ty $llvm_intr($args)",
    "
    $(@gen_ir("%a.jl.$i = extractvalue [$a_sz x $a_lc_ty] %0, $i", a_sz))
    $(@gen_ir("%b.jl.$i = extractvalue [$b_sz x $b_lc_ty] %1, $i", b_sz))
    $(@gen_ir("%c.jl.$i = extractvalue [$c_sz x $c_lc_ty] %2, $i", c_sz))

    $(@gen_ir("%a.llvm.$i = bitcast $a_lc_ty %a.jl.$i to $a_llvm_ty", a_sz))
    $(@gen_ir("%b.llvm.$i = bitcast $b_lc_ty %b.jl.$i to $b_llvm_ty", b_sz))
    $(@gen_ir("%c.llvm.$i = bitcast $c_lc_ty %c.jl.$i to $c_llvm_ty", c_sz))

    %d.llvm = call $d_struct_ty $llvm_intr($args)

    $(@gen_ir("%d.llvm.$i = extractvalue $d_struct_ty %d.llvm, $i", d_sz))

    $(@gen_ir("%d.jl.$i = bitcast $d_llvm_ty %d.llvm.$i to $d_lc_ty", d_sz))

    $(@gen_ir("%d.aggr.$i = insertvalue [$d_sz x $d_lc_ty] $(i == 0 ? "undef" : "%d.aggr.$(i-1)"), $d_lc_ty %d.jl.$i, $i", d_sz))

    ret [$d_sz x $d_lc_ty] %d.aggr.$(d_sz-1)
    ")

    @eval $func_name(a, b, c) = Base.llvmcall($ir,
        NTuple{$d_sz, $d_jl_ty},
        Tuple{NTuple{$a_sz, $a_jl_ty}, NTuple{$b_sz, $b_jl_ty}, NTuple{$c_sz, $c_jl_ty}},
        convert(NTuple{$a_sz, $a_jl_ty}, a),
        convert(NTuple{$b_sz, $b_jl_ty}, b),
        convert(NTuple{$c_sz, $c_jl_ty}, c))

    @eval export $func_name
end

################################################################################
# HIGH LEVEL (CUDA-STYLE API)
################################################################################

# -------------
# WMMA fragment
# -------------

export wmma_row_major, wmma_col_major, wmma_unspecified

abstract type wmma_fragment_layout end
struct wmma_row_major <: wmma_fragment_layout end
struct wmma_col_major <: wmma_fragment_layout end
struct wmma_unspecified <: wmma_fragment_layout end


export wmma_matrix_a, wmma_matrix_b, wmma_accumulator

abstract type wmma_fragment_use end
struct wmma_matrix_a <: wmma_fragment_use end
struct wmma_matrix_b <: wmma_fragment_use end
struct wmma_accumulator <: wmma_fragment_use end


export wmma_fragment

struct wmma_fragment{M, N, K, FS, T, L <: wmma_fragment_layout, U <: wmma_fragment_use}
    x::NTuple{FS, T}
end

# ------------------
# WMMA configuration
# ------------------

export wmma_config
struct wmma_config{M, N, K, d_type} end

# ---------
# Constants
# ---------

map_matrix_to_use = Dict(
                      "a" => wmma_matrix_a,
                      "b" => wmma_matrix_b,
                      "c" => wmma_accumulator,
                      "d" => wmma_accumulator
                        )

map_address_space_to_ty = Dict(
                               "" => AS.Generic,
                               "shared" => AS.Shared,
                               "global" => AS.Global
                              )

# ----------------
# Helper functions
# ----------------

get_matrix_use(mat) = map_matrix_to_use[mat]
get_address_space(as) = map_address_space_to_ty[as]

# ---------
# WMMA load
# ---------

export wmma_load_a, wmma_load_b, wmma_load_c

for mat in ["a", "b", "c"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]


    # Float32 is only supported for C
    if (elem_type == "f32") && (mat != "c")
        continue
    end

    # Name of Julia function
    func_name = Symbol("wmma_load_$mat")

    # Name of the Julia wrapper
    wrapper = Symbol(join_nonempty("llvm", "wmma", "load", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Get fragment size
    frag_sz = get_frag_sz(mat, elem_type)

    # Get Julia element type
    julia_type = get_jl_ty(mat, elem_type)

    # Get matrix use type
    matrix_use = get_matrix_use(mat)

    # Get layout type
    layout_ty = (layout == "col") ? wmma_col_major : wmma_row_major
    layout_frag_ty = (mat == "c") ? wmma_unspecified : layout_ty

    # Get pointer type
    ptr_ty = (elem_type == "f32") ? Float32 : Float16

    # Get address space type
    as_ty = get_address_space(addr_space)

    @eval function $func_name(addr::DevicePtr{$ptr_ty, $as_ty},
                              stride::Number,
                              layout::Type{$layout_ty},
                              config::Type{wmma_config{16, 16, 16, d_type}}) where d_type
        x = $wrapper(addr, stride)
        return wmma_fragment{16, 16, 16, $frag_sz, $julia_type, $layout_frag_ty, $matrix_use}(x)
    end
end


# ------------------------
# WMMA multiply-accumulate
# ------------------------

export wmma_mma

for a_layout in ["col", "row"],
    b_layout in ["col", "row"],
    shape in ["m16n16k16"],
    d_elem_type in ["f16", "f32"],
    c_elem_type in ["f16", "f32"],
    b_elem_type in ["f16"],
    a_elem_type in ["f16"]

    # Name of the Julia wrapper
    wrapper = Symbol(join_nonempty("llvm", "wmma", "mma", a_layout, b_layout, shape, d_elem_type, c_elem_type, "_"))

    # Information about a
    a_frag_sz = get_frag_sz("a", a_elem_type)
    a_julia_type = get_jl_ty("a", a_elem_type)
    a_layout_ty = (a_layout == "col") ? wmma_col_major : wmma_row_major

    # Information about b
    b_frag_sz = get_frag_sz("b", b_elem_type)
    b_julia_type = get_jl_ty("b", b_elem_type)
    b_layout_ty = (b_layout == "col") ? wmma_col_major : wmma_row_major

    # Information about c
    c_frag_sz = get_frag_sz("c", c_elem_type)
    c_julia_type = get_jl_ty("c", c_elem_type)

    # Information about d
    d_frag_sz = get_frag_sz("d", d_elem_type)
    d_julia_type = get_jl_ty("d", d_elem_type)

    # We need some way to select if we want d to be 16 or 32-bit floating point
    # during dispatch.
    dispatch_ty = (d_elem_type == "f16") ? Float16 : Float32

    @eval function wmma_mma(a::wmma_fragment{16, 16, 16, $a_frag_sz, $a_julia_type, $a_layout_ty, wmma_matrix_a},
                            b::wmma_fragment{16, 16, 16, $b_frag_sz, $b_julia_type, $b_layout_ty, wmma_matrix_b},
                            c::wmma_fragment{16, 16, 16, $c_frag_sz, $c_julia_type, wmma_unspecified, wmma_accumulator},
                            conf::Type{wmma_config{16, 16, 16, $dispatch_ty}})
        x = $wrapper(a.x, b.x, c.x)
        return wmma_fragment{16, 16, 16, $d_frag_sz, $d_julia_type, wmma_unspecified, wmma_accumulator}(x)
    end
end


# ----------
# WMMA store
# ----------

export wmma_store_d

for mat in ["d"],
    layout in ["col", "row"],
    shape in ["m16n16k16"],
    addr_space in ["", "shared", "global"],
    stride in ["stride"],
    elem_type in ["f16", "f32"]

    # Name of Julia function
    func_name = Symbol("wmma_store_$mat")

    # Name of the Julia wrapper
    wrapper = Symbol(join_nonempty("llvm", "wmma", "store", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Get fragment size
    frag_sz = get_frag_sz(mat, elem_type)

    # Get Julia element type
    julia_type = get_jl_ty(mat, elem_type)

    # Get matrix use type
    matrix_use = get_matrix_use(mat)

    # Get layout type
    layout_ty = (layout == "col") ? wmma_col_major : wmma_row_major
    layout_frag_ty = wmma_unspecified

    # Get pointer type
    ptr_ty = (elem_type == "f32") ? Float32 : Float16

    # Get address space type
    as_ty = get_address_space(addr_space)

    @eval function $func_name(addr::DevicePtr{$ptr_ty, $as_ty},
                              d::wmma_fragment{16, 16, 16, $frag_sz, $julia_type, $layout_frag_ty, $matrix_use},
                              stride::Number,
                              layout::Type{$layout_ty},
                              config::Type{wmma_config{16, 16, 16, d_type}}) where d_type
        $wrapper(addr, d.x, stride)
        return nothing
    end

end


# ------------------
# WMMA fill fragment
# ------------------

export wmma_fill_c

for mat in ["c"],
    elem_type in ["f16", "f32"]

    # Name of the Julia function
    func_name = Symbol("wmma_fill_$mat")

    # Get fragment size
    frag_sz = get_frag_sz(mat, elem_type)

    # Get Julia type
    julia_type = get_jl_ty(mat, elem_type)

    # Value type
    val_type = (elem_type == "f16") ? Float16 : Float32

    # Returned tuple
    if elem_type == "f16"
        tuple = :(ntuple(i -> ntuple(j -> VecElement{Float16}(value), 2), $frag_sz))
    else
        tuple = :(ntuple(i -> value, $frag_sz))
    end

    @eval function $func_name(value::$val_type)
        x = $tuple
        return wmma_fragment{16, 16, 16, $frag_sz, $julia_type, wmma_unspecified, wmma_accumulator}(x)
    end
end
