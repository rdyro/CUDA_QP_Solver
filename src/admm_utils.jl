@inline function admm_set_rhs_top!(temp, sig, x, q)
  @csimd for i in 1:length(temp)
    @cinbounds temp[i] = sig * x[i] - q[i]
  end
  return
end

@inline function admm_set_rhs_bot!(temp, z, y, rho0)
  @csimd for i in 1:length(temp)
    @cinbounds temp[i] = z[i] - y[i] / rho0
  end
  return
end

@inline function admm_update_z!(z, v, y, rho0)
  @csimd for i in 1:length(z)
    @cinbounds z[i] += (v[i] - y[i]) / rho0
  end
  return
end

@inline function admm_update_y!(y, z, l, u, rho0)
  @csimd for i in 1:length(y)
    new_z = max(min(z[i] + y[i] / rho0, u[i]), l[i])
    @cinbounds y[i] += rho0 * (z[i] - new_z)
    @cinbounds z[i] = new_z
  end
  return
end

####################################################################################################

@inline function admm_set_rhs_top!(temp, sig, x, q, sidx, eidx)
  @csimd for i in sidx:eidx
    @cinbounds temp[i] = sig * x[i] - q[i]
  end
  return
end

@inline function admm_set_rhs_bot!(temp, z, y, rho0, sidx, eidx)
  @csimd for i in sidx:eidx
    @cinbounds temp[i] = z[i] - y[i] / rho0
  end
  return
end

@inline function admm_update_z!(z, v, y, rho0, sidx, eidx)
  @csimd for i in sidx:eidx
    @cinbounds z[i] += (v[i] - y[i]) / rho0
  end
  return
end

@inline function admm_update_y!(y, z, l, u, rho0, sidx, eidx)
  @csimd for i in sidx:eidx
    new_z = max(min(z[i] + y[i] / rho0, u[i]), l[i])
    @cinbounds y[i] += rho0 * (z[i] - new_z)
    @cinbounds z[i] = new_z
  end
  return
end
