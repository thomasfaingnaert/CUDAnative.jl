export Layout
module Layout

using CUDAnative
using CUDAnative.Tiling
using GPUifyLoops
using StaticArrays

# -----------
# Layout base
# -----------

abstract type LayoutBase{T} end

@inline eltype(::Type{<:LayoutBase{T}}) where {T} = T
@inline size(::Type{<:LayoutBase{T}}, logical_size::NamedTuple) where {T} = Tuple(logical_size)

# --------------
# Padded layouts
# --------------

struct Padded{L, P} end

@inline function pad_logical_coord(::Type{Padded{L, P}}, crd::NamedTuple) where {L, P}
    t = Tuple(crd)
    return typeof(crd)((Base.first(t) + P, Base.tail(t)...))
end

@inline eltype(::Type{Padded{L, P}}) where {L, P} = eltype(L)
@inline size(::Type{Padded{L, P}}, logical_size::NamedTuple) where {L, P} = size(L, pad_logical_coord(Padded{L, P}, logical_size))
@inline load(::Type{Padded{L, P}}, workspace, tile::Tile, logical_size::NamedTuple) where {L, P} = load(L, workspace, tile, pad_logical_coord(Padded{L, P}, logical_size))
@inline store!(::Type{Padded{L, P}}, workspace, value, tile::Tile, logical_size::NamedTuple) where {L, P} = store!(L, workspace, value, tile::Tile, pad_logical_coord(Padded{L, P}, logical_size))

# ---------------
# AlignedColMajor
# ---------------

struct AlignedColMajor{T} <: LayoutBase{T} end

# Helper function to determine alignment from offset
@inline function get_alignment(base_alignment, offset)
    rem = offset % base_alignment

    if rem == 0
        return base_alignment # same alignment as base
    elseif rem & (rem - 1) == 0
        return rem            # partially aligned
    else
        return 1              # not aligned
    end
end

# TODO: cleanup vectorisation

@inline function load(::Type{AlignedColMajor{Float16}}, workspace, tile::Tile{size}, logical_size::NamedTuple) where {size}
    res = MArray{Tuple{size[1], size[2]}, Float16}(undef)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : size[1]
            t = translate(tile, (i - 1, j - 1))
            ind = Tuple(t.index) .+ 1
            @inbounds linear_index = LinearIndices(Base.size(workspace))[ind...]
            @inbounds res[i, j] = workspace[linear_index]
        end
    end

    return res
end

@inline function load(::Type{AlignedColMajor{Float32}}, workspace, tile::Tile{size}, logical_size::NamedTuple) where {size}
    res = MArray{Tuple{size[1], size[2]}, Float32}(undef)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : size[1]
            t = translate(tile, (i - 1, j - 1))
            ind = Tuple(t.index) .+ 1
            @inbounds linear_index = LinearIndices(Base.size(workspace))[ind...]

            #= align = Val(get_alignment(16 ÷ sizeof(T), i-1)) =#
            #= @inbounds res[i, j] = unsafe_load(pointer(workspace), linear_index) =#
            @inbounds res[i, j] = workspace[linear_index]
        end
    end

    return res
end

# TODO: remove logical_size?
@inline function store!(::Type{AlignedColMajor{T}}, workspace, value, tile::Tile{size}, logical_size::NamedTuple) where {T, size}
    @unroll for j = 1 : size[2]
        @unroll for i = 1 : size[1]
            t = translate(tile, (i - 1, j - 1))
            ind = Tuple(t.index) .+ 1
            @inbounds workspace[ind...] = value[i, j]
        end
    end
end

end
