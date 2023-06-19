using LinearAlgebra, SparseArrays, BenchmarkTools
using PyPlot, Printf
using CUDA, OSQP
#using Infiltrator, Profile, ProfileView
CUDA.allowscalar(false)

include(abspath(joinpath(@__DIR__, "..", "src", "vec.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "sputils.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "gpu_sparse.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "mpc.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "ldlt.jl")))
include(abspath(joinpath(@__DIR__, "..", "src", "qp.jl")))

################################################################################
unwrap(A) = (A.colptr, A.rowval, A.nzval)
################################################################################

# make the system #################
N, XDIM, UDIM = 100, 2, 1

function solve_routine(P, q, A, l, u; device = "cuda")
  n, m = length(q), length(l)
  sol = zeros(Float64, 2 * N * (XDIM + UDIM))
  iwork, fwork = zeros(Int64, 10^6), zeros(Float64, 10^6)
  Pp, Pi, Px = unwrap(P)
  Ap, Ai, Ax = unwrap(A)
  debug = false
  if lowercase(device) == "cuda"
    @time CUDA.@sync begin
      sol = CuArray{Float32}(sol)
      Pp, Pi, Px = CuArray{Int32}(Pp), CuArray{Int32}(Pi), CuArray{Float32}(Px)
      Ap, Ai, Ax = CuArray{Int32}(Ap), CuArray{Int32}(Ai), CuArray{Float32}(Ax)
      q = CuArray{Float32}(q)
      l, u = CuArray{Float32}(l), CuArray{Float32}(u)
      iwork, fwork = CuArray{Int32}(iwork), CuArray{Float32}(fwork)
    end
    for i in 1:10
      begin
        args = (sol, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, l, u, iwork, fwork, debug)
        kernel = @cuda launch=false QP_solve!(args...)
        #CUDA.@sync @cuda threads = 1 blocks = 1 QP_solve!(args...)
        config = launch_configuration(kernel.fun)
        begin
          CUDA.@profile kernel(args...; threads=1, blocks=1)
          secs = kernel(args...; threads=1, blocks=1)
          println(secs)
        end
        #CUDA.@sync @cuda threads = 1 blocks = 1 QP_solve!(args...)
      end
    end
    sol = Array(sol)
  else
    args = (sol, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, l, u, iwork, fwork, debug)
    #@time QP_solve!(args...)
    @time QP_solve!(args...)
    @time QP_solve!(args...)
    @time QP_solve!(args...)
    #Profile.clear()
    #@profile for i in 1:10
    for i in 1:10
      QP_solve!(args...)
    end
    #ProfileView.view()
  end
  return sol
end

function solve_osqp(P, q, A, b, l, u)
  m = OSQP.Model()
  OSQP.setup!(m, P = P, q = q, A = A, l = l, u = u, verbose = false)
  result = OSQP.solve!(m)
  for i in 1:10
    @time begin
      local m
      m = OSQP.Model()
      OSQP.setup!(m; P = P, q = q, A = A, l = l, u = u, verbose = false)
      result = OSQP.solve!(m)
    end
  end
  return copy(result.x)
end

function extract_vars(sol)
  u = reshape(view(sol, 1:N*UDIM), (UDIM, N))

  x = [x0 reshape(view(sol, N*UDIM+1:N*UDIM+N*XDIM), (XDIM, N))]
  return x, u
end

# plot solution
function plot_dynamics(x, u; clear = true, kw...)
  figure(324234)
  (clear) && (clf())
  plot(u[1, :]; label = "u1", alpha = 0.5, kw...)
  plot(x[1, :]; label = "x1", alpha = 0.5, kw...)
  plot(x[2, :]; label = "x2", alpha = 0.5, kw...)
  legend()
  tight_layout()
  return
end

###################################

x0 = Float64[5.0; 5.0]
fx = map(_ -> Float64[1.0 0.1; 0.0 1.0], 1:N)
fu = map(_ -> Float64[0.0; 1.0], 1:N)
f = map(i -> i == 1 ? fx[i] * x0 : zeros(Float64, 2), 1:N)
f, fx, fu = cat(f...; dims = 2), cat(fx...; dims = 3), cat(fu...; dims = 3)

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
l = [b; -0.3 * ones(Float64, N * UDIM)]
u = [b; 0.3 * ones(Float64, N * UDIM)]

#args = P, q, A, l, u
#plot_dynamics(extract_vars(solve_osqp(P, q, A, b, l, u))...; clear = true)
plot_dynamics(extract_vars(solve_routine(P, q, A, l, u; device = "cuda"))...)
#println("End")
