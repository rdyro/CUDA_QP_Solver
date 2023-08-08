using LinearAlgebra, SparseArrays, BenchmarkTools
using PyPlot, Printf, JuMP, OSQP
using CUDA, OSQP, Cthulhu, Infiltrator
#using Infiltrator, Profile, ProfileView
CUDA.allowscalar(false)
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

include(abspath(joinpath(@__DIR__, "..", "src", "vec.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "sputils.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "gpu_sparse.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "mpc.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "ldlt.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "qp_static.jl")))

################################################################################
sparseunwrap(A) = (A.colptr, A.rowval, A.nzval)
################################################################################

# make the system #################
N, XDIM, UDIM = 37, 2, 1

memory_mask = [zeros(Int32, 13); ones(Int32, 9)]
eval(define_kernel(memory_mask))
out = nothing

function solve_routine(P, q, A, l, u)
  info = zeros(Int32, 3)
  n, m = length(q), length(l)
  iwork, fwork = zeros(Int64, 10^6), zeros(Float64, 10^6)
  Pp, Pi, Px = sparseunwrap(P)
  Ap, Ai, Ax = sparseunwrap(A)

  @infiltrate

  iters = 200
  sol = CuArray{Float32}(zeros(n + m))
  info = CuArray{Int32}(info)
  Pp, Pi, Px = CuArray{Int32}(Pp), CuArray{Int32}(Pi), CuArray{Float32}(Px)
  Ap, Ai, Ax = CuArray{Int32}(Ap), CuArray{Int32}(Ai), CuArray{Float32}(Ax)
  q = CuArray{Float32}(q)
  l, u = CuArray{Float32}(l), CuArray{Float32}(u)
  iwork, fwork = CuArray{Int32}(iwork), CuArray{Float32}(fwork)

  # CUDA ##################################################
  #args = (sol, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, l, u)
  #args = (sol, n, m, P, q, A, l, u)

  args = (sol, info, iters, n, m, (Pp, Pi, Px), q, (Ap, Ai, Ax), l, u, iwork, fwork)
  bench = @benchmark CUDA.@sync @cuda threads=1 blocks=1 shmem=2^15 QP_solve!($args...)
  println("Times:")
  for t in bench.times
    println(t / 1e9)
  end
  println("Mean time: ", mean(bench.times) / 1e9)
  CUDA.@sync @cuda threads = 1 blocks = 1 shmem = 2^15 QP_solve!(args...)

  # CPU ###################################################
  #sol = Array(sol)
  #QP_solve!(sol, n, m, Array(Pp), Array(Pi), Array(Px), Array(q), Array(Ap), Array(Ai), Array(Ax), Array(l), Array(u))

  # actually solve ##############kk
  #CUDA.@sync @cuda threads=1 blocks=1 QP_solve!(args...)
  println("Max imem: ", Array(info)[1])
  println("Max fmem: ", Array(info)[2])
  println("Iterations: ", Array(info)[3])
  sol = Array(sol)
  return sol
end

function solve_osqp(P, q, A, l, u)
  m = OSQP.Model()
  OSQP.setup!(m, P=P, q=q, A=A, l=l, u=u, verbose=false)
  return copy(OSQP.solve!(m).x)
end

function solve_jump(P, q, A, l, u)
  model = Model(OSQP.Optimizer)
  JuMP.set_silent(model)
  @variable(model, z[1:length(q)])
  @constraint(model, A * z .<= u)
  @constraint(model, A * z .>= l)
  @objective(model, Min, 0.5 * dot(z, P, z) + dot(z, q))
  optimize!(model)
  return copy(value.(z))
end

function split_vars(sol)
  u = reshape(view(sol, 1:N*UDIM), (UDIM, N))
  x = [x0 reshape(view(sol, N*UDIM+1:N*UDIM+N*XDIM), (XDIM, N))]
  return x, u
end

# plot solution
function plot_dynamics(x, u; clear=true, kw...)
  figure(324234)
  (clear) && (clf())
  plot(u[1, :]; label="u1", alpha=0.5, kw...)
  plot(x[1, :]; label="x1", alpha=0.5, kw...)
  plot(x[2, :]; label="x2", alpha=0.5, kw...)
  legend()
  tight_layout()
  return
end

# main #############################################################################################

x0 = Float64[5.0; 5.0]
fx = map(_ -> Float64[1.0 0.1; 0.0 1.0], 1:N)
fu = map(_ -> Float64[0.0; 1.0], 1:N)
f = map(i -> i == 1 ? fx[i] * x0 : zeros(Float64, 2), 1:N)
f, fx, fu = cat(f...; dims=2), cat(fx...; dims=3), cat(fu...; dims=3)

# construct A b
Annz = (N - 1) * (XDIM + XDIM^2 + UDIM * XDIM) + XDIM * UDIM + XDIM
Ap, Ai = zeros(Int64, N * (XDIM + UDIM) + 1), zeros(Int64, Annz)
Ax, b = zeros(Float64, Annz), zeros(Float64, N * XDIM)
construct_Ab!(Ap, Ai, Ax, b, f, fx, fu)
A = SparseMatrixCSC(N * XDIM, N * (XDIM + UDIM), Ap, Ai, Ax)
A = [A; sparse(I, N * UDIM, N * UDIM) spzeros(N * UDIM, N * XDIM)]

# construct P q
P = spdiagm(0 => ones(Float64, N * (XDIM + UDIM)))
q = zeros(Float64, N * (XDIM + UDIM))

# construct limits
#inf = Float64(1e20)
#l = [-0.3 * ones(Float64, N * UDIM); -inf * ones(Float64, N * XDIM)]
#u = [0.3 * ones(Float64, N * UDIM); inf * ones(Float64, N * XDIM)]
l = [b; -1.0 * ones(Float64, N * UDIM)]
u = [b; 1.0 * ones(Float64, N * UDIM)]

x1, u1 = split_vars(solve_osqp(P, q, A, l, u))
#sol = solve_routine(P, q, A, l, u)
x2, u2 = split_vars(solve_routine(P, q, A, l, u))
x3, u3 = split_vars(solve_jump(P, q, A, l, u))

@printf("Error x = %.4e\n", norm(x1 - x2) / norm(x2))
@printf("Error u = %.4e\n", norm(u1 - u2) / norm(u2))
#println()
#@printf("Error x = %.4e\n", norm(x1 - x3) / norm(x3))
#@printf("Error u = %.4e\n", norm(u1 - u3) / norm(u3))
#println()
#@printf("Error x = %.4e\n", norm(x2 - x3) / norm(x3))
#@printf("Error u = %.4e\n", norm(u2 - u3) / norm(u3))

#plot(u2[1, :], label="u2")
#plot(u1[1, :], label="u1")
#legend()

nothing
