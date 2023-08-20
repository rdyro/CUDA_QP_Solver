using LinearAlgebra, SparseArrays, BenchmarkTools
using PyPlot, Printf, JuMP, OSQP, JSON
using CUDA, OSQP, Cthulhu, Infiltrator
using NVTX
CUDA.allowscalar(false)
BenchmarkTools.DEFAULT_PARAMETERS.samples = 100
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 5

include(abspath(joinpath(@__DIR__, "..", "src", "header.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "amd_pkg", "amd_pkg.jl")))

################################################################################
sparseunwrap(A) = (A.colptr, A.rowval, A.nzval)
################################################################################

function read_matrix(fname)
  data = JSON.parse(read(fname, String))
  Ap = Int.(data["Ap"])
  Ai = Int.(data["Ai"])
  Ax = Float64.(data["Ax"])
  m = maximum(Ai)
  n = length(Ap) - 1
  return SparseMatrixCSC{Float64,Int64}(m, n, Ap, Ai, Ax)
end

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
  :mem_perm => 1,
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
  :mem_y => 1,
  :mem_v => 1,
  :use_amd => 1,
)

eval(define_kernel_cuda(config))
eval(define_kernel_cpu(config))
function to_cuda(x)
  if typeof(x) <: Tuple
    return map(to_cuda, x)
  else
    return eltype(x) <: AbstractFloat ? CuArray{Float32}(x) : CuArray{Int32}(x)
  end
end
#to_cuda(x) = CUDA.cu(x)
to_cpu(x) = Array(x)

copy_tuple(x) = map(copy, x)

function solve_routine(P, q, A, l, u)
  info = zeros(Int32, 5)
  n, m = length(q), length(l)
  iwork, fwork = zeros(Int32, 10^6), zeros(Float32, 10^6)
  Pp, Pi, Px = sparseunwrap(P)
  Ap, Ai, Ax = sparseunwrap(A)

  iters = 200
  sol = zeros(n + m)

  # CUDA ##################################################
  #sol, info, Pp, Pi, Px, Ap, Ai, Ax, q, l, u, iwork, fwork = map(to_cuda, vars)

  args = (sol, info, iters, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, l, u, iwork, fwork)
  n_blocks, n_threads = 164, 64
  args = map(x -> reduce(vcat, [copy(x) for _ in 1:n_blocks]), args)
  sols, infos, iterss, ns, ms, Pps, Pis, Pxs, qs, Aps, Ais, Axs, ls, us, iwork, fwork = args
  work_sizes = fill(Int32(10^6), n_blocks)
  n_offsets = Int32.(cumsum([0; fill(n, n_blocks)]))[1:end-1]
  m_offsets = Int32.(cumsum([0; fill(m, n_blocks)]))[1:end-1]
  Pnnz_offsets = Int32.(cumsum([0; fill(Pp[end] - 1, n_blocks)]))[1:end-1]
  Annz_offsets = Int32.(cumsum([0; fill(Ap[end] - 1, n_blocks)]))[1:end-1]
  work_offsets = Int32.(cumsum([0; fill(work_sizes[1], n_blocks)]))[1:end-1]

  offsets = (n_offsets, m_offsets, Pnnz_offsets, Annz_offsets, work_offsets)
  args = (
    sols,
    infos,
    iterss,
    ns,
    ms,
    (Pps, Pis, Pxs),
    qs,
    (Aps, Ais, Axs),
    ls,
    us,
    iwork,
    fwork,
    work_sizes,
    offsets,
  )
  args = map(to_cuda, args)
  sols,
  infos,
  iterss,
  ns,
  ms,
  (Pps, Pis, Pxs),
  qs,
  (Aps, Ais, Axs),
  ls,
  us,
  iwork,
  fwork,
  work_sizes,
  offsets = args

  CUDA.@sync @cuda threads = n_threads blocks = n_blocks shmem = 2^15 QP_solve_cuda!(
    args...,
    n_threads,
  )

  bench =
    @benchmark CUDA.@sync @cuda threads = $n_threads blocks = $n_blocks shmem = 2^15 QP_solve_cuda!(
      $args...,
      $n_threads,
    )
  display(bench)
  sol_out = copy(sols[1:n+m])

  # actually solve ##############kk
  #CUDA.@sync @cuda threads=1 blocks=1 QP_solve!(args...)
  println("Max imem_fast: ", Array(infos)[1])
  println("Max fmem_fast: ", Array(infos)[2])
  println("Max imem_slow: ", Array(infos)[3])
  println("Max fmem_slow: ", Array(infos)[4])
  println("Iterations: ", Array(infos)[5])
  return Array(sol_out)
end

function solve_routine_cpu(P, q, A, l, u)
  info = zeros(Int32, 5)
  n, m = length(q), length(l)
  iwork, fwork = zeros(Int32, 10^6), zeros(Float32, 10^6)
  Pp, Pi, Px = sparseunwrap(P)
  Ap, Ai, Ax = sparseunwrap(A)

  iters = 200
  sol = zeros(n + m)

  # CUDA ##################################################
  #sol, info, Pp, Pi, Px, Ap, Ai, Ax, q, l, u, iwork, fwork = map(to_cuda, vars)

  args = (sol, info, iters, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, l, u, iwork, fwork)
  n_blocks, n_threads = 164, 64
  args = map(x -> reduce(vcat, [copy(x) for _ in 1:n_blocks]), args)
  sols, infos, iterss, ns, ms, Pps, Pis, Pxs, qs, Aps, Ais, Axs, ls, us, iwork, fwork = args
  work_sizes = fill(Int32(10^6), n_blocks)
  n_offsets = Int32.(cumsum([0; fill(n, n_blocks)]))[1:end-1]
  m_offsets = Int32.(cumsum([0; fill(m, n_blocks)]))[1:end-1]
  Pnnz_offsets = Int32.(cumsum([0; fill(Pp[end] - 1, n_blocks)]))[1:end-1]
  Annz_offsets = Int32.(cumsum([0; fill(Ap[end] - 1, n_blocks)]))[1:end-1]
  work_offsets = Int32.(cumsum([0; fill(work_sizes[1], n_blocks)]))[1:end-1]

  offsets = (n_offsets, m_offsets, Pnnz_offsets, Annz_offsets, work_offsets)
  args = (
    sols,
    infos,
    iterss,
    ns,
    ms,
    (Pps, Pis, Pxs),
    qs,
    (Aps, Ais, Axs),
    ls,
    us,
    iwork,
    fwork,
    work_sizes,
    offsets,
  )
  sols,
  infos,
  iterss,
  ns,
  ms,
  (Pps, Pis, Pxs),
  qs,
  (Aps, Ais, Axs),
  ls,
  us,
  iwork,
  fwork,
  work_sizes,
  offsets = args

  QP_solve_cpu!(args...)

  bench = @benchmark QP_solve_cpu!($args...)
  display(bench)
  sol_out = copy(sols[1:n+m])

  # actually solve ##############kk
  #CUDA.@sync @cuda threads=1 blocks=1 QP_solve!(args...)
  println("Max imem_fast: ", Array(infos)[1])
  println("Max fmem_fast: ", Array(infos)[2])
  println("Max imem_slow: ", Array(infos)[3])
  println("Max fmem_slow: ", Array(infos)[4])
  println("Iterations: ", Array(infos)[5])
  return Array(sol_out)
end

function solve_osqp(P, q, A, l, u)
  m = OSQP.Model()
  OSQP.setup!(
    m,
    P=P,
    q=q,
    A=A,
    l=l,
    u=u;
    verbose=true,
    polishing=false,
    adaptive_rho=false,
    check_termination=0,
    linsys_solver="qdldl",
    max_iter=200,
    rho=1e1,
  )
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
u_lim = 1.0
l = [b; -u_lim * ones(Float64, N * UDIM)]
u = [b; u_lim * ones(Float64, N * UDIM)]

x1, u1 = split_vars(solve_osqp(P, q, A, l, u))
x2, u2 = split_vars(solve_routine(P, q, A, l, u))
x3, u3 = split_vars(solve_routine_cpu(P, q, A, l, u))
x4, u4 = split_vars(solve_jump(P, q, A, l, u))

println()
@printf("OSQP error x = %.4e\n", norm(x1 - x4) / norm(x4))
@printf("OSQP error u = %.4e\n", norm(u1 - u4) / norm(u4))
println()
@printf("CUDA error x = %.4e\n", norm(x2 - x4) / norm(x4))
@printf("CUDA error u = %.4e\n", norm(u2 - u4) / norm(u4))
println()
@printf("CPU error x =  %.4e\n", norm(x3 - x4) / norm(x4))
@printf("CPU error u =  %.4e\n", norm(u3 - u4) / norm(u4))

nothing
