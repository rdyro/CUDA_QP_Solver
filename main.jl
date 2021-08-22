include("ldlt.jl")
include("mpc.jl")
include("vec.jl")
include("gpu_sparse.jl")
include("qp.jl")

using LinearAlgebra, SparseArrays, BenchmarkTools
using PyPlot, Printf
using CUDA, OSQP

################################################################################
unwrap(A) = (A.colptr, A.rowval, A.nzval)
################################################################################

CUDA.allowscalar(false)

DEVICE = "cuda"

# make the system #################
x0 = Float64[5.0; 5.0]
N, xdim, udim = 100, 2, 1
fx = map(_ -> Float64[1.0 0.1; 0.0 1.0], 1:N)
fu = map(_ -> Float64[0.0; 1.0], 1:N)
f = map(i -> i == 1 ? fx[i] * x0 : zeros(Float64, 2), 1:N)
f, fx, fu = cat(f...; dims = 2), cat(fx...; dims = 3), cat(fu...; dims = 3)

Annz = (N - 1) * (xdim + xdim^2 + udim * xdim) + xdim * udim + xdim
Ap, Ai = zeros(Int64, N * (xdim + udim) + 1), zeros(Int64, Annz)
Ax, b = zeros(Float64, Annz), zeros(Float64, N * xdim)
construct_Ab!(Ap, Ai, Ax, b, f, fx, fu)

P = spdiagm(0 => ones(Float64, N * (xdim + udim)))
Pp, Pi, Px = unwrap(P)
Pp, Pi, Px = Array{Int64}(Pp), Array{Int64}(Pi), Array{Float64}(Px)
q = zeros(Float64, N * (xdim + udim))
inf = Float64(1e20)
l = [-ones(Float64, N * udim); -inf * ones(Float64, N * xdim)]
u = [ones(Float64, N * udim); inf * ones(Float64, N * xdim)]
sol = zeros(Float64, N * (xdim + udim + xdim))
iwork, fwork = zeros(Int64, 10^6), zeros(Float64, 10^6)
n, m = N * (xdim + udim), N * xdim

if lowercase(DEVICE) == "cuda"
  @time CUDA.@sync begin
    sol = CuArray{Float32}(sol)
    Pp, Pi, Px = CuArray{Int32}(Pp), CuArray{Int32}(Pi), CuArray{Float32}(Px)
    Ap, Ai, Ax = CuArray{Int32}(Ap), CuArray{Int32}(Ai), CuArray{Float32}(Ax)
    q, b = CuArray{Float32}(q), CuArray{Float32}(b)
    l, u = CuArray{Float32}(l), CuArray{Float32}(u)
    iwork, fwork = CuArray{Int32}(iwork), CuArray{Float32}(fwork)
  end

  for i in 1:10
    @time begin
      args = (sol, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, b, l, u, iwork, fwork)
      CUDA.@sync @cuda threads = 1 blocks = 1 QP_solve!(args...)
    end
  end
  sol = Array(sol)
else
  for i in 1:10
    @time QP_solve!(sol, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, b, l, u, iwork, fwork)
    println()
  end
end

P = SparseMatrixCSC(n, n, Array(Pp), Array(Pi), Array(Px))
A = SparseMatrixCSC(m, n, Array(Ap), Array(Ai), Array(Ax))
q, b, l, u =
  Array{Float64}(q), Array{Float64}(b), Array{Float64}(l), Array{Float64}(u)
G = sparse(I, n, n)
Aosqp = [A; G]
losqp = [b; l]
uosqp = [b; u]

# plot solution
u = reshape(view(sol, 1:N*udim), (udim, N))
x = [x0 reshape(view(sol, N*udim+1:N*udim+N*xdim), (xdim, N))]

figure(1)
clf()
plot(u[1, :]; label = "u1", alpha = 0.5)
plot(x[1, :]; label = "x1", alpha = 0.5)
plot(x[2, :]; label = "x2", alpha = 0.5)
legend()
tight_layout()

m = OSQP.Model()
OSQP.setup!(m, P = P, q = q, A = Aosqp, l = losqp, u = uosqp, verbose = false)
result = OSQP.solve!(m)
for i in 1:10
  @time begin
    local m
    m = OSQP.Model()
    OSQP.setup!(
      m;
      P = P,
      q = q,
      A = Aosqp,
      l = losqp,
      u = uosqp,
      verbose = false,
    )
    result = OSQP.solve!(m)
  end
  println()
end

sol = result.x
u = reshape(view(sol, 1:N*udim), (udim, N))
x = [x0 reshape(view(sol, N*udim+1:N*udim+N*xdim), (xdim, N))]

plot(u[1, :]; label = "u1", ls = ":", alpha = 0.5)
plot(x[1, :]; label = "x1", ls = ":", alpha = 0.5)
plot(x[2, :]; label = "x2", ls = ":", alpha = 0.5)
legend()
tight_layout()
###################################

println("End")
