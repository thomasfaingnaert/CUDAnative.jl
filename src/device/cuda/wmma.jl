llvm_wmma_load_a_row_m16n16k16_stride_f16(src_addr, stride) = ccall("extern llvm.nvvm.wmma.m16n16k16.load.a.row.stride.f16.p0i8", llvmcall, NTuple{8, NTuple{2, VecElement{Float16}}}, (Ref{Float16}, Int32), src_addr, stride)

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
