function unflatten_recurse(typ::Type{NTuple{8, VecElement{Float16}}}, e, idx)
    ret = Expr(:tuple)

    for (i, eltyp) in enumerate(typ.types)
        arg = :(VecElement{Float16}($e[$idx]))
        push!(ret.args, arg)
    end

    return ret, idx
end

@generated unflatten(::Type{typ}, x) where typ = unflatten_recurse(typ, :x, 1)[1]
