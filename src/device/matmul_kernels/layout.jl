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

# TODO: readd vectorisation
@inline function load(::Type{AlignedColMajor{T}}, workspace, tile::Tile{size}, logical_size::NamedTuple) where {T, size}
    res = MArray{Tuple{size[1], size[2]}, T}(undef)

    @unroll for j = 1 : size[2]
        @unroll for i = 1 : size[1]
            t = translate(tile, (i - 1, j - 1))
            @inbounds res[i, j] = workspace[linearise(t.index, logical_size)]
        end
    end

    return res
end

# TODO: remove logical_size?
@inline function store!(::Type{AlignedColMajor{T}}, workspace, value, tile::Tile{size}, logical_size::NamedTuple) where {T, size}
    @unroll for j = 1 : size[2]
        @unroll for i = 1 : size[1]
            t = translate(tile, (i - 1, j - 1))
            @inbounds workspace[linearise(t.index, logical_size)] = value[i, j]
        end
    end
end

end
