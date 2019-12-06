################################################################################
# CONSTANTS
################################################################################

# Maps PTX types to Julia array types
map_ptx_to_jl_array = Dict(
                           "f16" => Float16,
                           "f32" => Float32
                          )

# Maps PTX types to Julia fragment types
map_ptx_to_jl_frag = Dict(
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

# Maps PTX AS to Int
map_ptx_as_to_int = Dict(
                         "" => 0,
                         "shared" => 3,
                         "global" => 1
                        )

################################################################################
# HELPER FUNCTIONS
################################################################################

function join_nonempty(args...)
    delim = args[end]
    arr = [args[1:end-1]...]

    return join(arr[arr .!= ""], delim)
end

# Returns (Julia array type, Julia fragment type, fragment size)
get_frag_info(matrix, ptx_el_type) = (
        map_ptx_to_jl_array[ptx_el_type],
        map_ptx_to_jl_frag[ptx_el_type],
        map_frag_sizes["$matrix.$ptx_el_type"]
        )

get_addrspace_info(addr_space) = map_ptx_as_to_int[addr_space]

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

    addr_space_int = get_addrspace_info(addr_space)

    # Name of the Julia wrapper function
    func_name = Symbol(join_nonempty("llvm", "wmma", "load", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.load.$mat.$layout.stride.$elem_type.p$(addr_space_int)i8"

    # Determine types + size for this (matrix, elem_type) combination
    arr_ty, frag_ty, sz = get_frag_info(mat, elem_type)

    ccall_name = "extern $llvm_intr"

    @eval $func_name(src_addr, stride) = ccall($ccall_name, llvmcall, NTuple{$sz, $frag_ty}, (Ref{$arr_ty}, Int32), src_addr, stride)
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

    addr_space_int = get_addrspace_info(addr_space)

    # Name of the Julia wrapper function
    func_name = Symbol(join_nonempty("llvm", "wmma", "store", mat, layout, shape, addr_space, stride, elem_type, "_"))

    # Name of the LLVM intrinsic
    llvm_intr = "llvm.nvvm.wmma.$shape.store.$mat.$layout.stride.$elem_type.p$(addr_space_int)i8"

    # Determine types + size for this (matrix, elem_type) combination
    arr_ty, frag_ty, sz = get_frag_info(mat, elem_type)

    ccall_name = "extern $llvm_intr"
    frag_types = ntuple(i -> frag_ty, sz)
    frag_vars = ntuple(i -> :(data[$i]), sz)

    @eval $func_name(dst_addr, data, stride) = ccall($ccall_name, llvmcall, Nothing, (Ref{$arr_ty}, $(frag_types...), Int32), dst_addr, $(frag_vars...), stride)
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
    llvm_intr = "llvm.nvvm.wmma.$shape.mma.$a_layout.$b_layout.$d_elem_type.$c_elem_type"

    # Determine types + size for the (matrix, elem_type) combinations for matrix A, B, C and D
    a_arr_ty, a_frag_ty, a_sz = get_frag_info("a", a_elem_type)
    b_arr_ty, b_frag_ty, b_sz = get_frag_info("b", b_elem_type)
    c_arr_ty, c_frag_ty, c_sz = get_frag_info("c", c_elem_type)
    d_arr_ty, d_frag_ty, d_sz = get_frag_info("d", d_elem_type)

    ccall_name = "extern $llvm_intr"

    a_types = ntuple(i -> a_frag_ty, a_sz)
    b_types = ntuple(i -> b_frag_ty, b_sz)
    c_types = ntuple(i -> c_frag_ty, c_sz)

    a_vars = ntuple(i -> :(a[$i]), a_sz)
    b_vars = ntuple(i -> :(b[$i]), b_sz)
    c_vars = ntuple(i -> :(c[$i]), c_sz)

    @eval $func_name(a, b, c) = ccall($ccall_name, llvmcall, NTuple{$d_sz, $d_frag_ty}, ($(a_types...), $(b_types...), $(c_types...)), $(a_vars...), $(b_vars...), $(c_vars...))
    @eval export $func_name
end

################################################################################
# FLATTENING/UNFLATTENING LOGIC
################################################################################

# Base case (Float16, Float32, ...)
flatten_recurse(typ, e) = [:($e)]
unflatten_recurse(typ, e, idx) = :($e[$idx]), idx + 1

# VecElements
flatten_recurse(typ::Type{VecElement{T}}, e) where T = [:($e.value)]
unflatten_recurse(typ::Type{VecElement{T}}, e, idx) where T = :(VecElement{$T}($e[$idx])), idx + 1

# NTuples
function flatten_recurse(typ::Type{NTuple{N, T}}, e) where {N, T}
    ret = Expr[]

    for (i, eltyp) in enumerate(typ.types)
        append!(ret, flatten_recurse(eltyp, :($e[$i])))
    end

    return ret
end

function unflatten_recurse(typ::Type{NTuple{N, T}}, e, idx) where {N, T}
    ret = Expr(:tuple)

    for (i, eltyp) in enumerate(typ.types)
        arg, idx = unflatten_recurse(eltyp, e, idx)
        push!(ret.args, arg)
    end

    return ret, idx
end

@generated flatten(x::typ) where typ = Expr(:tuple, flatten_recurse(typ, :x)...)
@generated unflatten(::Type{typ}, x) where typ = unflatten_recurse(typ, :x, 1)[1]
