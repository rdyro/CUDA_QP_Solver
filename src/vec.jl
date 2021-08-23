# Routines #####################################################################
@inline function veccpy!(c, a)
  #= @inbounds =# for i in 1:length(a)
    c[i] = a[i] 
  end
  return
end

@inline function vecadd!(c, a)
  #= @inbounds =# for i in 1:length(a)
    c[i] = c[i] + a[i] 
  end
  return
end

@inline function vecsub!(c, a)
  #= @inbounds =# for i in 1:length(a)
    c[i] = c[i] - a[i] 
  end
  return
end

@inline function vecdot(c, a)
  res = 0.0
  #= @inbounds =# for i in 1:length(a)
    res += c[i] * a[i]
  end
  return res
end

@inline function vecscal!(c, alf)
  #= @inbounds =# for i in 1:length(c)
    c[i] = c[i] * alf
  end
  return
end

@inline function vecfill!(c, alf)
  #= @inbounds =# for i in 1:length(c)
    c[i] = alf
  end
  return
end

@inline function vecclamp!(c, a, l, u)
  #= @inbounds =# for i in 1:length(a)
    c[i] = a[i] > u[i] ? u[i] : (a[i] < l[i] ? l[i] : a[i])
  end
  return
end
# Routines #####################################################################
