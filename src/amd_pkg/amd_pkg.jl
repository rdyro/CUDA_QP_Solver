module AMDPKG

using Printf, LinearAlgebra, SparseArrays, Statistics, Random

macro identity(x) return esc(x) end
var"@cinbounds" = var"@inbounds"
#var"@cinbounds" = var"@identity"

Vec{T} = AbstractVector{T} where {T}
SparseTuple = Tuple{Vec, Vec, Vec}

include(joinpath(@__DIR__, "sort_utils.jl"))
include(joinpath(@__DIR__, "ordering.jl"))
include(joinpath(@__DIR__, "permute.jl"))

export permute_matrix
export find_ordering, compute_fillin

end