module Operator

using CUDAnative: WMMA
using CUDAnative.Tiling: Tile, linearise
using CUDAnative.MatMul.Layout: Padded, AlignedColMajor, pad_logical_coord

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
