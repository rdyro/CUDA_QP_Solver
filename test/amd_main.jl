using LinearAlgebra, SparseArrays, BenchmarkTools
using PyPlot, Printf, JuMP, OSQP
using CUDA, OSQP, Cthulhu, Infiltrator
#using Infiltrator, Profile, ProfileView
CUDA.allowscalar(false)
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

include(abspath(joinpath(@__DIR__, "..", "src", "vec_utils.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "mem_utils.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "gpu_sparse.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "admm_utils.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "mpc.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "ldlt.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "qp_cpu.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "qp_cuda.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "amd_pkg", "amd_pkg.jl")))

################################################################################
sparseunwrap(A) = (A.colptr, A.rowval, A.nzval)
################################################################################

# make the system #################
N, XDIM, UDIM = 37, 2, 1

config = Dict{Symbol,Int32}(
  :mem_AT => 0,
  :mem_A => 0,
  :mem_P => 0,
  :mem_q => 0,
  :mem_lu => 0,
  :mem_H => 0,
  :mem_I => 0,
  :mem_T => 0,
  :mem_perm => 0,
  :mem_perm_work => 0,
  :mem_H_perm_work => 0,
  :mem_H_perm => 0,
  :mem_Lnz => 0,
  :mem_etree => 0,
  :mem_etree_iwork => 0,
  :mem_ldlt_iwork => 0,
  :mem_ldlt_fwork => 0,
  :mem_L => 1,
  :mem_D => 1,
  :mem_temp => 1,
  :mem_x => 1,
  :mem_z => 1,
  :mem_zproj => 1,
  :mem_y => 1,
  :mem_v => 1,
  :use_amd => 1,
)

eval(define_kernel_cuda(config))
eval(define_kernel_cpu(config))
to_cuda(x) = eltype(x) <: AbstractFloat ? CuArray{Float32}(x) : CuArray{Int32}(x)
to_cpu(x) = Array(x)

function solve_routine(P, q, A, l, u)
  info = zeros(Int32, 3)
  n, m = length(q), length(l)
  iwork, fwork = zeros(Int32, 10^6), zeros(Float32, 10^6)
  Pp, Pi, Px = sparseunwrap(P)
  Ap, Ai, Ax = sparseunwrap(A)

  iters = 200
  sol = zeros(n + m)

  # CUDA ##################################################
  vars = sol, info, Pp, Pi, Px, Ap, Ai, Ax, q, l, u, iwork, fwork
  sol, info, Pp, Pi, Px, Ap, Ai, Ax, q, l, u, iwork, fwork = map(to_cuda, vars)

  args = (sol, info, iters, n, m, (Pp, Pi, Px), q, (Ap, Ai, Ax), l, u, iwork, fwork)
  bench = @benchmark CUDA.@sync @cuda threads = 1 blocks = 1 shmem = 2^15 QP_solve_cuda!($args...)
  println("Times:")
  for t in bench.times
    println(t / 1e9)
  end
  println("Mean time: ", mean(bench.times) / 1e9)
  CUDA.@sync @cuda threads = 1 blocks = 1 shmem = 2^15 QP_solve_cuda!(args...)
  sol_out = copy(sol)

  # actually solve ##############kk
  #CUDA.@sync @cuda threads=1 blocks=1 QP_solve!(args...)
  println("Max imem: ", Array(info)[1])
  println("Max fmem: ", Array(info)[2])
  println("Iterations: ", Array(info)[3])
  return Array(sol_out)
end

function solve_routine_cpu(P, q, A, l, u)
  info = zeros(Int32, 3)
  n, m = length(q), length(l)
  iwork, fwork = zeros(Int32, 10^6), zeros(Float32, 10^6)
  Pp, Pi, Px = sparseunwrap(P)
  Ap, Ai, Ax = sparseunwrap(A)

  iters = 200
  sol = zeros(n + m)

  # CPU ###################################################
  vars = sol, info, Pp, Pi, Px, Ap, Ai, Ax, q, l, u, iwork, fwork
  sol, info, Pp, Pi, Px, Ap, Ai, Ax, q, l, u, iwork, fwork = map(to_cpu, vars)
  args = (sol, info, iters, n, m, (Pp, Pi, Px), q, (Ap, Ai, Ax), l, u, iwork, fwork)
  display(@benchmark QP_solve_cpu!($args...))

  QP_solve_cpu!(args...)
  return copy(sol)
end

function solve_osqp(P, q, A, l, u)
  m = OSQP.Model()
  #OSQP.setup!(m, P=P, q=q, A=A, l=l, u=u; verbose=true, polishing=false, adaptive_rho=false, max_iter=Int32(1e3), check_termination=0)
  OSQP.setup!(m, P=P, q=q, A=A, l=l, u=u; verbose=true, polishing=false, adaptive_rho=false, check_termination=0, linsys_solver="qdldl", cg_max_iter=10000)
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
x2, u2 = split_vars(solve_routine(P, q, A, l, u))
#x3, u3 = split_vars(solve_jump(P, q, A, l, u))
x4, u4 = split_vars(solve_routine_cpu(P, q, A, l, u))

@printf("Error x = %.4e\n", norm(x2 - x1) / norm(x1))
@printf("Error u = %.4e\n", norm(u2 - u1) / norm(u1))
println()
@printf("Error x = %.4e\n", norm(x4 - x1) / norm(x1))
@printf("Error u = %.4e\n", norm(u4 - u1) / norm(u1))

#plot(u2[1, :], label="u2")
#plot(u1[1, :], label="u1")
#legend()

nothing
