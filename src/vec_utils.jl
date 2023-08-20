@inline function veccpy!(c, a)
  @csimd for i in 1:length(a)
    @cinbounds c[i] = a[i]
  end
  return
end

@inline function vecadd!(c, a)
  @csimd for i in 1:length(a)
    @cinbounds c[i] = c[i] + a[i]
  end
  return
end

@inline function vecsub!(c, a)
  @csimd for i in 1:length(a)
    @cinbounds c[i] = c[i] - a[i]
  end
  return
end

@inline function vecdot(c, a)
  res = 0.0f0
  @csimd for i in 1:length(a)
    @cinbounds res += c[i] * a[i]
  end
  return res
end

@inline function vecscal!(c, alf)
  @csimd for i in 1:length(c)
    @cinbounds c[i] = c[i] * alf
  end
  return
end

@inline function vecfill!(c, alf)
  @csimd for i in 1:length(c)
    @cinbounds c[i] = alf
  end
  return
end

@inline function vecclamp!(c, a, l, u)
  @csimd for i in 1:length(a)
    #@cinbounds c[i] = a[i] >= u[i] ? u[i] : (a[i] <= l[i] ? l[i] : a[i])
    @cinbounds c[i] = max(min(a[i], u[i]), l[i])
  end
  return
end

@inline function vecpermute!(c, a, perm)
  @csimd for i in 1:length(a)
    @cinbounds c[i] = a[perm[i]]
  end
  return
end

####################################################################################################

@inline function veccpy!(c, a, sidx, eidx)
  @csimd for i in sidx:eidx
    @cinbounds c[i] = a[i]
  end
  return
end

@inline function vecpermute!(c, a, perm, sidx, eidx)
  @csimd for i in sidx:eidx
    @cinbounds c[i] = a[perm[i]]
  end
  return
end
