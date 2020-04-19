module Layout

using CUDAnative: Vec, vloada, vstorea!
using CUDAnative.Tiling: Tile, linearise

# -----------
# Layout base
# -----------

export LayoutBase, layout_eltype, layout_size, layout_load, layout_store!

abstract type LayoutBase{T} end

@inline layout_eltype(::Type{<:LayoutBase{T}}) where {T} = T
@inline layout_size(::Type{<:LayoutBase{T}}, logical_size::NamedTuple) where {T} = Tuple(logical_size)

# --------------
# Padded layouts
# --------------

export Padded
struct Padded{L, P} end

@inline function pad_logical_coord(::Type{Padded{L, P}}, crd::NamedTuple) where {L, P}
    t = Tuple(crd)
    return typeof(crd)((Base.first(t) + P, Base.tail(t)...))
end

@inline layout_eltype(::Type{Padded{L, P}}) where {L, P} = layout_eltype(L)
@inline layout_size(::Type{Padded{L, P}}, logical_size::NamedTuple) where {L, P} = layout_size(L, pad_logical_coord(Padded{L, P}, logical_size))
@inline layout_load(::Type{Padded{L, P}}, workspace, tile::Tile, logical_size::NamedTuple) where {L, P} = layout_load(L, workspace, tile, pad_logical_coord(Padded{L, P}, logical_size))
@inline layout_store!(::Type{Padded{L, P}}, workspace, value, tile::Tile, logical_size::NamedTuple) where {L, P} = layout_store!(L, workspace, value, tile::Tile, pad_logical_coord(Padded{L, P}, logical_size))

# ---------------
# AlignedColMajor
# ---------------

export AlignedColMajor
struct AlignedColMajor{T} <: LayoutBase{T} end

@inline function layout_load(::Type{AlignedColMajor{T}}, workspace, tile::Tile, logical_size::NamedTuple) where {T}
    N = 16 ÷ sizeof(T)
    ptr = pointer(workspace, linearise(tile.base, logical_size))
    return vloada(Vec{N, T}, ptr, linearise(tile.offset, logical_size))
end

@inline function layout_store!(::Type{AlignedColMajor{T}}, workspace, value, tile::Tile, logical_size::NamedTuple) where {T}
    N = 16 ÷ sizeof(T)
    ptr = pointer(workspace, linearise(tile.base, logical_size))
    return vstorea!(Vec{N, T}, ptr, value, linearise(tile.offset, logical_size))
end

end
