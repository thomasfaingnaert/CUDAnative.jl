################################################################################
# TILING API
################################################################################

module Tiling

# -----
# Tiles
# -----

export Tile
"""
    Tile{names, T}

A [`Tile`](@ref) represents a part of a multidimensional tensor that is
contiguous and aligned to the tensor's dimensions.

Note that the dimensions of this [`Tile`](@ref) are named. Similar to a
[`NamedTuple`](@ref), the names are stored as a type parameter `names`.

A [`Tile`](@ref) contains several fields:
- `index`: A [`NamedTuple`](@ref) that represents the "first" multidimensional
           index of the parent tensor that this tile contains.
- `base`: The part of the `index` that depends on runtime values, such as the
          `threadIdx`.
- `offset`: The part of the `index` that is known at compile-time.
- `size`: A [`NamedTuple`](@ref) representing the size of the tile along each
          dimension.

You can also project a [`Tile`](@ref) (i.e. drop certain dimensions) by
accessing a special "field" of which the name is derived from the dimensions
you intend to keep.

For example, to drop the `K` dimension of a tile containing `M`, `N` and `K`
dimensions, you can use the syntax `tile.MN`.
"""
struct Tile{names, T}
    base::NamedTuple{names, T}
    offset::NamedTuple{names, T}
    size::NamedTuple{names, T}
end

function Base.show(io::IO, tile::Tile{names, T}) where {names, T}
    print(io, "base:   ", tile.base, '\n')
    print(io, "offset: ", tile.offset, '\n')
    print(io, "size:   ", tile.size)
end

"""
    Tile(; kw_args...)

Creates a new [`Tile`](@ref) of the given `size`, with zero `base` and
`offset`. The `size` for each dimension must be specified by a keyword
argument.

# Example
```julia
CUDAnative.Tiling.Tile(M = 24, N = 16, K = 4)
```
"""
Tile(; kw_args...) = Tile((; kw_args...))

"""
    Tile(size::NamedTuple{names, T})

Creates a new [`Tile`](@ref) of the given `size`, with zero `base` and
`offset`.

# Arguments
- `size`: A `NamedTuple` representing the size of the [`Tile`](@ref).

# Example
```julia
CUDAnative.Tiling.Tile((M = 24, N = 16, K = 4))
```
"""
Tile(size::NamedTuple{names, T}) where {names, T} = Tile{names, T}(map(x -> 0, size), map(x -> 0, size), size)

@generated function getproperty_impl(tile::Tile{names, T}, ::Val{sym}) where {names, T, sym}
    if sym == :base || sym == :offset || sym == :size
        # fields
        return :(getfield(tile, sym))
    elseif sym == :index
        # index: sum of base and offset
        return :(map(+, getfield(tile, :base), getfield(tile, :offset)))
    else
        # tile projection
        sym_str = String(sym)
        names = ntuple(i -> Symbol(sym_str[i]), length(sym_str))
        return :( Tile(NamedTuple{$names}(getfield(tile, :base)), NamedTuple{$names}(getfield(tile, :offset)), NamedTuple{$names}(getfield(tile, :size))) )
    end
end

@inline Base.getproperty(tile::Tile{names, T}, sym::Symbol) where {names, T} = getproperty_impl(tile, Val(sym))

export linearise

"""
    linearise(coord::NamedTuple{names, T}, dims::NamedTuple{names, T})

Convert a multidimensional coordinate to a linear index with respect to a
tensor with dimensions `dims`.

# Arguments
- `coord`: A `NamedTuple` representing the coordinate.
- `dims`: A `NamedTuple` representing the size of the parent tensor.
"""
@inline function linearise(coord::NamedTuple{names, T}, dims::NamedTuple{names, T}) where {names, T}
    ind = Tuple(coord) .+ 1
    @inbounds return LinearIndices(Tuple(dims))[ind...]
end

export translate

"""
    translate(tile::Tile{names, T}, offset::NamedTuple{names, T})

Translate (i.e. move) a [`Tile`](@ref) by a constant `offset`.

# Arguments
- `tile`: The [`Tile`](@ref) to translate.
- `offset`: The `offset` in each dimension.
"""
@inline function translate(tile::Tile{names, T}, offset::NamedTuple{names, T}) where {names, T}
    base = map(+, tile.base, offset)
    return Tile(base, tile.offset, tile.size)
end

# -------------
# TileIterators
# -------------

export TileIterator

"""
    TileIterator{names, T, N, R}

A [`TileIterator`](@ref) represents an iterator over a set of [`Tile`](@ref)s.

See also: [`subdivide`](@ref), [`parallellise`](@ref).
"""
struct TileIterator{names, T, N, R}
    parent::Tile{names, T}
    tile_size::T
    subtile_indices::CartesianIndices{N, R}
    idx::Int32
    step::Int32
end

export parallellise

"""
    parallellise(tile, tiling_size, idx, size)

Split the given `tile` in subtiles of size `tiling_size` across a group of
cooperating entities (e.g. warps, threads, ...).

Unlike [`subdivide`](@ref), the `tile` need not be completely covered by
`count` tiles of size `tiling_size`. If that's not the case, the subtiles
are evenly parallellised across all cooperating entities.

Returns a [`TileIterator`](@ref) that iterates over the [`Tile`](@ref)s of
the calling entity.

# Arguments
- `tile`: The [`Tile`](@ref) to parallellise.
- `tiling_size`: A `NamedTuple` indicating the size of a subtile along each dimension.
- `idx`: The identity of the calling entity.
- `count`: The number of cooperating entities.
"""
@inline function parallellise(tile::Tile{names, T}, tiling_size::NamedTuple{names, T}, idx, count) where {names, T}
    # Number of tiles along each dimension
    num_tiles = map(div, Tuple(tile.size), Tuple(tiling_size))

    parent = tile
    tile_size = Tuple(tiling_size)
    subtile_indices = CartesianIndices(num_tiles)
    step = count

    return TileIterator(parent, tile_size, subtile_indices, convert(Int32, idx), convert(Int32, step))
end

export subdivide

"""
    subdivide(tile, tiling_size, idx, count)

Split the given `tile` in subtiles of size `tiling_size` across a group of
`count` cooperating entities (e.g. warps, threads, ...).

The given `tile` must be completely covered by `count` tiles of size
`tiling_size`.

Returns the [`Tile`](@ref) that the calling entity is responsible for.

# Arguments
- `tile`: The [`Tile`](@ref) to subdivide.
- `tiling_size`: A `NamedTuple` indicating the size of a subtile along each dimension.
- `idx`: The identity of the calling entity.
- `count`: The number of cooperating entities.
"""
@inline function subdivide(tile::Tile{names, T}, tiling_size::NamedTuple{names, T}, idx, count) where {names, T}
    return iterate(parallellise(tile, tiling_size, idx, count))[1]
end

@inline function Base.iterate(it::TileIterator{names, T, N, R}, state = 1) where {names, T, N, R}
    if state > length(it.subtile_indices)
        return nothing
    end

    # Calculate base and offset in number of tiles
    @inbounds base   = Tuple(it.parent.base)   .+ (Tuple(it.subtile_indices[it.idx]) .- 1) .* Tuple(it.tile_size)
    @inbounds offset = Tuple(it.parent.offset) .+ (Tuple(it.subtile_indices[state])  .- 1) .* Tuple(it.tile_size)

    # Create tile
    tile = Tile{names, T}(NamedTuple{names, T}(base), NamedTuple{names, T}(offset), NamedTuple{names, T}(it.tile_size))

    return (tile, state + it.step)
end

end


################################################################################
# GEMM API
################################################################################

module Gemm

################################################################################
# GEMM API: LAYOUTS
################################################################################

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

################################################################################
# GEMM API: OPERATORS
################################################################################

module Operator

using CUDAnative: WMMA
using CUDAnative.Tiling: Tile, linearise
using CUDAnative.Gemm.Layout: Padded, AlignedColMajor

export op_shape
export op_fragtype_a, op_fragtype_b, op_fragtype_accum
export op_load_a, op_load_b, op_load_c, op_store_d
export op_mma

# ----
# WMMA
# ----

export WMMAOp
struct WMMAOp{M, N, K} end

op_shape(::Type{WMMAOp{M, N, K}}) where {M, N, K} = (M = M, N = N, K = K)

op_fragtype_a(::Type{WMMAOp{16, 16, 16}}, ::Type{Padded{AlignedColMajor{Float16}, PADDING}}) where {PADDING} = WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixA}
op_fragtype_b(::Type{WMMAOp{16, 16, 16}}, ::Type{Padded{AlignedColMajor{Float16}, PADDING}}) where {PADDING} = WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixB}
op_fragtype_accum(::Type{WMMAOp{16, 16, 16}}, ::Type{AlignedColMajor{Float32}}) = WMMA.Fragment{16, 16, 16, 8, Float32, WMMA.Unspecified, WMMA.Accumulator}

function op_load_a(::Type{WMMAOp{M, N, K}}, ::Type{Padded{AlignedColMajor{Float16}, PADDING}}, workspace, tile::Tile, logical_size::NamedTuple) where {M, N, K, PADDING}
    t = Tuple(logical_size)
    padded_logical_size = typeof(logical_size)((Base.first(t) + PADDING, Base.tail(t)...))

    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, padded_logical_size))
    return WMMA.load_a(ptr, padded_logical_size.M, WMMA.ColMajor, conf)
end

function op_load_b(::Type{WMMAOp{M, N, K}}, ::Type{Padded{AlignedColMajor{Float16}, PADDING}}, workspace, tile::Tile, logical_size::NamedTuple) where {M, N, K, PADDING}
    t = Tuple(logical_size)
    padded_logical_size = typeof(logical_size)((Base.first(t) + PADDING, Base.tail(t)...))

    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, padded_logical_size))
    return WMMA.load_b(ptr, padded_logical_size.K, WMMA.ColMajor, conf)
end

function op_load_c(::Type{WMMAOp{M, N, K}}, ::Type{AlignedColMajor{Float32}}, workspace, tile::Tile, logical_size::NamedTuple) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, logical_size))
    return WMMA.load_c(ptr, logical_size.M, WMMA.ColMajor, conf)
end

function op_store_d(::Type{WMMAOp{M, N, K}}, ::Type{AlignedColMajor{Float32}}, workspace, frag, tile::Tile, logical_size::NamedTuple) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, logical_size))
    WMMA.store_d(ptr, frag, logical_size.M, WMMA.ColMajor, conf)
end

function op_mma(::Type{WMMAOp{M, N, K}}, a_frag, b_frag, c_frag) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    return WMMA.mma(a_frag, b_frag, c_frag, conf)
end



end

end
