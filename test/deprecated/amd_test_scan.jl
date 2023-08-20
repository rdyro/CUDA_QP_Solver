using Random, Statistics, LinearAlgebra, SparseArrays, Printf, JSON
using AMD, BenchmarkTools
using .Threads: @threads
using ProgressBars
using PyPlot
include(joinpath(@__DIR__, "..", "src", "amd_pkg", "amd_pkg.jl"))

find_ordering, compute_fillin = AMDPKG.find_ordering, AMDPKG.compute_fillin

data_file = joinpath(@__DIR__, "..", "data", "all_matrices.json")
all_matrices = JSON.parse(read(data_file, String))
function find_ordering_(Ap, Ai; method="3depth")
  n = length(Ap) - 1
  perm = zeros(Int, n)
  iwork = zeros(Int, n * 10)
  find_ordering(perm, Ap, Ai, iwork; method=method)
  return perm
end

results = Dict(
  "id" => Int[],
  "random" => Int[],
  "amd" => Int[],
  "own_5depth" => Int[],
  "own_3depth" => Int[],
  "own_frozen" => Int[],
  "own_amd" => Int[],
)
@threads for matrix in ProgressBar(all_matrices)
  A = SparseMatrixCSC(
    matrix["n"],
    matrix["n"],
    Int.(matrix["colptr"]),
    Int.(matrix["rowval"]),
    randn(length(matrix["rowval"])),
  )
  A = A + A'
  n = size(A, 1)
  Ap, Ai = copy(A.colptr), copy(A.rowval)
  iwork = zeros(Int, n * 10)
  push!(results["id"], compute_fillin(Ap, Ai, 1:n, iwork))
  push!(
    results["own_3depth"],
    compute_fillin(Ap, Ai, find_ordering_(Ap, Ai, method=:depth3), iwork),
  )
  push!(
    results["own_5depth"],
    compute_fillin(Ap, Ai, find_ordering_(Ap, Ai, method=:depth5), iwork),
  )
  push!(
    results["own_frozen"],
    compute_fillin(Ap, Ai, find_ordering_(Ap, Ai, method=:frozen), iwork),
  )
  push!(results["own_amd"], compute_fillin(Ap, Ai, find_ordering_(Ap, Ai, method=:amd), iwork))
  push!(results["amd"], compute_fillin(Ap, Ai, amd(A), iwork))
  rand_val = round(Int, median([compute_fillin(Ap, Ai, randperm(n), iwork) for _ in 1:50]))
  push!(results["random"], rand_val)
end

vals = map(
  x -> x ./ results["id"],
  [
    results["own_3depth"],
    results["own_5depth"],
    results["own_frozen"],
    results["own_amd"],
    results["amd"],
  ],
)

figure()
hist(vals, label=["3depth", "5depth", "frozen", "amd", "AMD"], range=(0.0, 5.0), bins=20)
legend()
show()

@printf("3depth = %.2f%%\n", mean(vals[1] .< 1.0) * 1e2)
@printf("5depth = %.2f%%\n", mean(vals[2] .< 1.0) * 1e2)
@printf("frozen = %.2f%%\n", mean(vals[3] .< 1.0) * 1e2)
@printf("amd =    %.2f%%\n", mean(vals[4] .< 1.0) * 1e2)
@printf("AMD =    %.2f%%\n", mean(vals[5] .< 1.0) * 1e2)
