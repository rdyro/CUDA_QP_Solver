using Random, Statistics, LinearAlgebra, SparseArrays, Printf
using AMD, BenchmarkTools
using .Threads: @threads
using ProgressBars
using PyPlot
include(joinpath(@__DIR__, "..", "src", "amd_pkg", "amd_pkg.jl"))

find_ordering, compute_fillin = AMDPKG.find_ordering, AMDPKG.compute_fillin


perms = []
for i in 1:1
  n = 300
  A = sprandn(n, n, 1e-2)
  n_bits = div(n, sizeof(Int) * 8) + 1
  #A = speye(n)
  A = A + A' + 1e-5 * I
  @assert norm(A' - A) / norm(A) < 1e-9
  display(A)
  Ap, Ai = A.colptr, A.rowval
  inbuilt = nnz(ldlt(A))
  seq = nnz(ldlt(A; perm=1:size(A, 1)))
  amd_ = nnz(ldlt(A; perm=AMD.amd(A)))
  symamd_ = nnz(ldlt(A; perm=AMD.symamd(A)))
  for method in [:depth3, :frozen]
    perm = zeros(Int, size(A, 1))
    iwork = zeros(Int, 4 * n + 3 * n_bits)
    find_ordering(perm, Ap, Ai, iwork; method=method)
    #bench = @benchmark find_ordering($perm, $Ap, $Ai, $iwork, $config)
    push!(perms, copy(perm))
    #bench = @benchmark find_ordering($perm, $Ap, $Ai, $iwork, $config);
    #display(bench)
    @assert length(unique(perm)) == length(perm)
    own = nnz(ldlt(A; perm=perm))
    inv_own = nnz(ldlt(A; perm=sortperm(perm)))
    @printf("AMD is     %07.3f%% of seq\n", amd_ / seq * 100)
    @printf("inbuilt is %07.3f%% of seq\n", inbuilt / seq * 100)
    @printf("own is     %07.3f%% of seq\n", own / seq * 100)
    @printf("inv_own is %07.3f%% of seq\n", inv_own / seq * 100)
    @printf("SYMAMD is  %07.3f%% of seq\n", symamd_ / seq * 100)
    println()
    @printf("Regression %07.3f%%\n", own / inbuilt * 1e2)
    println()
    #display(bench)
  end
end