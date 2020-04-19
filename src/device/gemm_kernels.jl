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

export Gemm
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
using CUDAnative.Gemm.Layout: Padded, AlignedColMajor, pad_logical_coord

export op_shape
export op_fragtype_a, op_fragtype_b, op_fragtype_accum
export op_load_a, op_load_b, op_load_c, op_store_d
export op_mma

# -------------------------------------
# Default definition for padded layouts
# -------------------------------------

# Fragment types
for f in (:op_fragtype_a, :op_fragtype_b, :op_fragtype_accum)
    @eval @inline $f(op, ::Type{Padded{L, P}}, args...) where {L, P} = $f(op, L, args...)
end

# Load fragments
for f in (:op_load_a, :op_load_b, :op_load_c)
    @eval @inline $f(op, ::Type{Padded{L, P}}, workspace, tile::Tile, logical_size::NamedTuple) where {L, P} = $f(op, L, workspace, tile, pad_logical_coord(Padded{L, P}, logical_size))
end

# Store fragments
@inline op_store_d(op, ::Type{Padded{L, P}}, workspace, frag, tile::Tile, logical_size::NamedTuple) where {L, P} = op_store_d(op, L, workspace, frag, tile, pad_logical_coord(Padded{L, P}, logical_size))

# ----
# WMMA
# ----

export WMMAOp
struct WMMAOp{M, N, K} end

@inline op_shape(::Type{WMMAOp{M, N, K}}) where {M, N, K} = (M = M, N = N, K = K)

@inline op_fragtype_a(::Type{WMMAOp{16, 16, 16}}, ::Type{AlignedColMajor{Float16}}) = WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixA}
@inline op_fragtype_b(::Type{WMMAOp{16, 16, 16}}, ::Type{AlignedColMajor{Float16}}) = WMMA.Fragment{16, 16, 16, 16, Float16, WMMA.ColMajor, WMMA.MatrixB}
@inline op_fragtype_accum(::Type{WMMAOp{16, 16, 16}}, ::Type{AlignedColMajor{Float32}}) = WMMA.Fragment{16, 16, 16, 8, Float32, WMMA.Unspecified, WMMA.Accumulator}

function op_load_a(::Type{WMMAOp{M, N, K}}, ::Type{AlignedColMajor{Float16}}, workspace, tile::Tile, logical_size::NamedTuple) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, logical_size))
    return WMMA.load_a(ptr, logical_size.M, WMMA.ColMajor, conf)
end

function op_load_b(::Type{WMMAOp{M, N, K}}, ::Type{AlignedColMajor{Float16}}, workspace, tile::Tile, logical_size::NamedTuple) where {M, N, K}
    conf = WMMA.Config{M, N, K, Float32}
    ptr = pointer(workspace, linearise(tile.index, logical_size))
    return WMMA.load_b(ptr, logical_size.K, WMMA.ColMajor, conf)
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

################################################################################
# GEMM API: TRANSFORMS
################################################################################

module Transform

# ---------------------
# Elementwise transform
# ---------------------

export ElementwiseTransform

"""
    ElementwiseTransform{F}

A simple transformation that applies a function elementwise.

# Example
```julia
double_elements = ElementwiseTransform(x -> x * 2)
```
"""
struct ElementwiseTransform{F}
    func::F
end

@inline ElementwiseTransform() = ElementwiseTransform(identity)

@inline (transf::ElementwiseTransform)(x, tile) = transf.func.(x)

end

################################################################################
# GEMM API: CONFIG & KERNELS
################################################################################

using CUDAnative.Gemm.Layout: AlignedColMajor, Padded
using CUDAnative.Gemm.Operator: WMMAOp, op_shape

struct Config{
    #= Params =#
    GEMM_SHAPE,                 # MNK, overall shape of the GEMM operation
    BLOCK_SHAPE,                # MNK, shape of each CTA tile
    WARPS_PER_BLOCK,            # scalar, number of warps per CTA

    MEM_A_WARP,                 # MK, shape of each warp tile during memory operations involving matrix A
    MEM_A_THREAD,               # MK, shape of each thread tile during memory operations involving matrix A

    MEM_B_WARP,                 # KN, shape of each warp tile during memory operations involving matrix B
    MEM_B_THREAD,               # KN, shape of each thread tile during memory operations involving matrix B

    MEM_CD_WARP,                # MN, shape of each warp tile during memory operations involving matrix C or D
    MEM_CD_THREAD,              # MN, shape of each thread tile during memory operations involving matrix C or D

    COMPUTE_WARP,               # MNK, shape of each warp tile during the inner loop computations
    COMPUTE_OP_SHAPE,           # MNK, shape of the operation used in the inner loop

    #= Layouts =#
    GLOBAL_A_LAYOUT,            # layout of the A matrix in global memory
    GLOBAL_B_LAYOUT,            # layout of the B matrix in global memory
    GLOBAL_C_LAYOUT,            # layout of the C matrix in global memory
    GLOBAL_D_LAYOUT,            # layout of the D matrix in global memory

    SHARED_A_LAYOUT,            # layout of the A matrix in shared memory
    SHARED_B_LAYOUT,            # layout of the B matrix in shared memory
    SHARED_C_LAYOUT,            # layout of the C matrix in shared memory
    SHARED_D_LAYOUT,            # layout of the D matrix in shared memory

    #= Operator =#
    OPERATOR,                   # which operator to use in the inner loop
   }
end

@inline function Base.getproperty(conf::Type{Config{GEMM_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR}}, sym::Symbol) where {GEMM_SHAPE, BLOCK_SHAPE, WARPS_PER_BLOCK, MEM_A_WARP, MEM_A_THREAD, MEM_B_WARP, MEM_B_THREAD, MEM_CD_WARP, MEM_CD_THREAD, COMPUTE_WARP, COMPUTE_OP_SHAPE, GLOBAL_A_LAYOUT, GLOBAL_B_LAYOUT, GLOBAL_C_LAYOUT, GLOBAL_D_LAYOUT, SHARED_A_LAYOUT, SHARED_B_LAYOUT, SHARED_C_LAYOUT, SHARED_D_LAYOUT, OPERATOR}
    if sym == :launch_args
        return (threads = WARPS_PER_BLOCK * 32, blocks = (GEMM_SHAPE.M ÷ BLOCK_SHAPE.M, GEMM_SHAPE.N ÷ BLOCK_SHAPE.N), shmem = 64 * 1024)
    else
        return getfield(conf, sym)
    end
end

function get_config(; gemm_shape, kwargs...)
    params = Dict(kwargs)

    return Config{
        #= Params =#
        gemm_shape,
        get(params, :block_shape, (M = 128, N = 128, K = 64)),
        get(params, :warps_per_block, 8),
        get(params, :mem_a_warp, (M = 128, K = 2)),
        get(params, :mem_a_thread, (M = 8, K = 1)),
        get(params, :mem_b_warp, (K = 64, N = 4)),
        get(params, :mem_b_thread, (K = 8, N = 1)),
        get(params, :mem_cd_warp, (M = 128, N = 1)),
        get(params, :mem_cd_thread, (M = 4, N = 1)),
        get(params, :compute_warp, (M = 32, N = 64, K = 16)),
        op_shape(get(params, :operator, WMMAOp{16, 16, 16})),

        #= Layouts =#
        get(params, :global_a_layout, AlignedColMajor{Float16}),
        get(params, :global_b_layout, AlignedColMajor{Float16}),
        get(params, :global_c_layout, AlignedColMajor{Float32}),
        get(params, :global_d_layout, AlignedColMajor{Float32}),

        get(params, :shared_a_layout, Padded{AlignedColMajor{Float16}, 8}),
        get(params, :shared_b_layout, Padded{AlignedColMajor{Float16}, 8}),
        get(params, :shared_c_layout, AlignedColMajor{Float32}),
        get(params, :shared_d_layout, AlignedColMajor{Float32}),

        #= Operators =#
        get(params, :operator, WMMAOp{16, 16, 16}),
    }
end

end
