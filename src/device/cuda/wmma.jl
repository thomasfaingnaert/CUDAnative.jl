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

# Name of the Julia wrapper function
func_name = Symbol(join_nonempty("llvm", "wmma", "load", "a", "row", "m16n16k16", "stride", "f16", "_"))

# Name of the LLVM intrinsic
llvm_intr = "llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i8"

ccall_name = "extern $llvm_intr"

@eval $func_name(src_addr, stride) = ccall($ccall_name, llvmcall, NTuple{8, NTuple{2, VecElement{Float16}}}, (Ref{Float16}, Int32), src_addr, stride)
@eval export $func_name

################################################################################
# FLATTENING/UNFLATTENING LOGIC
################################################################################

# Base case (Float16, Float32, ...)
unflatten_recurse(typ, e, idx) = :($e[$idx]), idx + 1

# VecElements
unflatten_recurse(typ::Type{VecElement{T}}, e, idx) where T = :(VecElement{$T}($e[$idx])), idx + 1

function unflatten_recurse(typ::Type{NTuple{N, T}}, e, idx) where {N, T}
    ret = Expr(:tuple)

    for (i, eltyp) in enumerate(typ.types)
        arg, idx = unflatten_recurse(eltyp, e, idx)
        push!(ret.args, arg)
    end

    return ret, idx
end

@generated unflatten(::Type{typ}, x) where typ = unflatten_recurse(typ, :x, 1)[1]
