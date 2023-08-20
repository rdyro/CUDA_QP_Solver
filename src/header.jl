macro identity(x)
  return esc(x)
end

var"@cinbounds" = var"@inbounds"
#var"@cinbounds" = var"@identity"
#var"@csimd" = var"@simd"
var"@csimd" = var"@identity"

include(joinpath(@__DIR__, "mem_utils.jl"))
include(joinpath(@__DIR__, "vec_utils.jl"))
include(joinpath(@__DIR__, "sparse_utils.jl"))
include(joinpath(@__DIR__, "admm_utils.jl"))
include(joinpath(@__DIR__, "ldlt.jl"))
include(joinpath(@__DIR__, "qp_cpu.jl"))
include(joinpath(@__DIR__, "qp_cuda.jl"))
include(joinpath(@__DIR__, "mpc.jl"))
