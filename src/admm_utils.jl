@inline function admm_set_rhs_top!(temp, sig, x, q)
  @simd for i in 1:length(temp)
    @cinbounds temp[i] = sig * x[i] - q[i]
  end
  return
end

@inline function admm_set_rhs_bot!(temp, z, y, rho0)
  @simd for i in 1:length(temp)
    @cinbounds temp[i] = z[i] - y[i] / rho0
  end
  return
end

@inline function admm_update_z!(z, v, y, rho0)
  @simd for i in 1:length(z)
    @cinbounds z[i] += (v[i] - y[i]) / rho0
  end
  return
end

@inline function admm_update_y!(y, z, l, u, rho0)
  @simd for i in 1:length(y)
    @cinbounds y[i] += rho0 * (z[i] - max(min(z[i] + y[i] / rho0, u[i]), l[i]))
  end
  return
end
