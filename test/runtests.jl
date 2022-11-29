using LinearAlgebra, SparseArrays, Random, Printf
using Test
include(abspath(joinpath(@__DIR__, "..", "src", "gpu_sparse.jl")))

@testset "Sparse Linear Algebra Routines - binary search" begin
  @test binary_search([0, 1, 2, 3, 4], 1) == 2
  @test binary_search([0, 1, 3, 4], 2) == -1 # not found
  @test binary_search([0, 1, 2, 3, 4], 4) == 5
  @test binary_search([0, 1, 2, 3, 4], 5) == -1 # not found
  @test binary_search([0, 1, 2, 3, 4], -2) == -1 # not found
end

function test_mergesort(N)
  @assert N >= 50
  iwork = zeros(N)
  a = rand(1:100, N)
  lo = rand(10:N-20)
  hi = rand(lo+1:N-10)
  al, ah = copy(a[1:lo-1]), copy(a[hi+1:end])
  a_ = copy(a)
  sort!(a_, lo, hi, QuickSort, Base.Order.Forward)
  mergesort!(a, lo, hi, iwork)
  @test all(al .== a[1:lo-1]) && all(ah .== a[hi+1:end])
  @test issorted(a[lo:hi])
  if !all(a[lo:hi] .== a_[lo:hi])
    display(vcat(a[lo:hi]', a_[lo:hi]'))
  end

  #display(hcat(a[lo:hi], a_[lo:hi]))
  return
end

@testset "Sparse Linear Algebra Routines - mergesort!" begin
  for i in 1:10000
    test_mergesort(rand(100:10000))
  end
end

@testset "Sparse Linear Algebra Routines - sptriu!/sptril!" begin
  N = 100
  A = sprandn(N, N, 0.1)
  Au, Al = UpperTriangular(copy(A)), LowerTriangular(copy(A))

  Ap, Ai, Ax = copy(A.colptr), copy(A.rowval), copy(A.nzval)
  sptriu!(Ap, Ai, Ax)
  @test norm(Au - Array(SparseMatrixCSC(N, N, Ap, Ai, Ax))) == 0.0

  Ap, Ai, Ax = copy(A.colptr), copy(A.rowval), copy(A.nzval)
  sptril!(Ap, Ai, Ax)
  @test norm(Al - Array(SparseMatrixCSC(N, N, Ap, Ai, Ax))) == 0.0
end

################################################################################
function test_spmul(M, N)
  A = sprandn(M, N, 0.1)
  y, x = zeros(M), randn(N)

  Ap, Ai, Ax = copy(A.colptr), copy(A.rowval), copy(A.nzval)
  spmul!(y, Ap, Ai, Ax, x)
  err = norm(y - A * x) / norm(x)
  @test err < 1e-15
  return
end

@testset "Sparse Linear Algebra Routines - spmul!" begin
  for i in 1:100
    M, N = rand(20:500), rand(20:500)
    test_spmul(M, N)
  end
end

################################################################################
function test_spmatadd(M, N)
  A = sprandn(M, N, 0.1)
  B = sprandn(M, N, 0.1)
  Ap, Ai, Ax = copy(A.colptr), copy(A.rowval), copy(A.nzval)
  Bp, Bi, Bx = copy(B.colptr), copy(B.rowval), copy(B.nzval)

  Cp, Ci = zeros(Int, length(Ap)), zeros(Int, length(Ai) + length(Bi))
  Cx = zeros(length(Ax) + length(Bx))

  spmatadd!(Cp, Ci, Cx, Ap, Ai, Ax, Bp, Bi, Bx)
  C = SparseMatrixCSC(M, N, Cp, Ci, Cx)
  @test norm(C - (A + B)) < 1e-15
  return
end

@testset "Sparse Linear Algebra Routines - spmatadd!" begin
  for i in 1:100
    M, N = rand(20:500), rand(20:500)
    test_spmatadd(M, N)
  end
end


################################################################################
function test_spmatmul(M, N)
  A = sprandn(N, M, 0.1)
  B = sprandn(M, N, 0.1)
  Ap, Ai, Ax = copy(A.colptr), copy(A.rowval), copy(A.nzval)
  Bp, Bi, Bx = copy(B.colptr), copy(B.rowval), copy(B.nzval)

  n = 100 * (length(Ax) + length(Bx))
  Cp, Ci, Cx = zeros(Int, N + 1), zeros(Int, n), zeros(n)
  iwork = zeros(Int, n)

  spmatmul(Cp, Ci, Cx, N, Ap, Ai, Ax, Bp, Bi, Bx, iwork)
  Ci, Cx = Ci[1:Cp[N+1]-1], Cx[1:Cp[N+1]-1]
  C = SparseMatrixCSC(N, N, Cp, Ci, Cx)
  err = norm(C - (A * B)) / norm(A * B)
  @test err < 1e-15
  return
end

@testset "Sparse Linear Algebra Routines - spmatmul!" begin
  using Random
  Random.seed!(round(Int, time() * 1e6))
  for i in 1:1000
    begin
      M, N = rand(20:500), rand(20:500)
      test_spmatmul(M, N)
    end
    #test_spmatmul(10, 10)
  end
end

################################################################################
@testset "Sparse Linear Algebra Routines - spvcat!/sphcat!" begin
  M, N = 7, 10
  A = sprandn(M, N, 0.1)
  B = sprandn(M, N, 0.1)
  Ap, Ai, Ax = copy(A.colptr), copy(A.rowval), copy(A.nzval)
  Bp, Bi, Bx = copy(B.colptr), copy(B.rowval), copy(B.nzval)

  n = 2 * (length(Ax) + length(Bx))
  Cp, Ci, Cx = zeros(Int, N + N + 1), zeros(Int, n), zeros(n)

  sphcat!(Cp, Ci, Cx, Ap, Ai, Ax, Bp, Bi, Bx)
  C = SparseMatrixCSC(M, N + N, Cp, Ci, Cx)
  @test norm(C - hcat(A, B)) < 1e-15

  spvcat!(Cp, Ci, Cx, M, Ap, Ai, Ax, Bp, Bi, Bx)
  C = SparseMatrixCSC(M + M, N, Cp, Ci, Cx)
  @test norm(C - vcat(A, B)) < 1e-15
end

################################################################################
@testset "Sparse Linear Algebra Routines - sptranspose!" begin
  M, N = 7, 10
  A = sprandn(M, N, 0.1)
  Ap, Ai, Ax = copy(A.colptr), copy(A.rowval), copy(A.nzval)

  n = length(Ax)
  Cp, Ci, Cx = zeros(Int, M + 1), zeros(Int, n), zeros(n)

  sptranspose!(Cp, Ci, Cx, Ap, Ai, Ax)
  C = SparseMatrixCSC(N, M, Cp, Ci, Cx)
  @test norm(C - A') < 1e-15
end

################################################################################
@testset "Sparse Linear Algebra Routines - spdiagadd!" begin
  N = 10
  @assert mod(N, 2) == 0
  A = sprandn(N, N, 0.1)
  A[div(N, 2)+1:end, div(N, 2)+1:end] += sparse(3.0 * I, div(N, 2), div(N, 2))
  Ap, Ai, Ax = copy(A.colptr), copy(A.rowval), copy(A.nzval)
  n = length(Ax)

  alf = 1.21423
  spdiagadd!(Ap, Ai, Ax, div(N, 2) + 1, N, alf)
  C = SparseMatrixCSC(N, N, Ap, Ai, Ax)
  @test norm(
    C - (A + spdiagm(0 => vcat(zeros(div(N, 2)), alf * ones(div(N, 2))))),
  ) < 1e-15
end

################################################################################

return
