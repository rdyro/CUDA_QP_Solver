# Routines #####################################################################
@inline function veccpy!(c, a)
  for i in 1:length(a)  #= @inbounds =#
    c[i] = a[i]
  end
  return
end

@inline function vecadd!(c, a)
  for i in 1:length(a)  #= @inbounds =#
    c[i] = c[i] + a[i]
  end
  return
end

@inline function vecsub!(c, a)
  for i in 1:length(a)  #= @inbounds =#
    c[i] = c[i] - a[i]
  end
  return
end

@inline function vecdot(c, a)
  res = 0f0
  for i in 1:length(a)  #= @inbounds =#
    res += c[i] * a[i]
  end
  return res
end

@inline function vecscal!(c, alf)
  for i in 1:length(c)  #= @inbounds =#
    c[i] = c[i] * alf
  end
  return
end

@inline function vecfill!(c, alf)
  for i in 1:length(c)  #= @inbounds =#
    c[i] = alf
  end
  return
end

@inline function vecclamp!(c, a, l, u)
  for i in 1:length(a)  #= @inbounds =#
    c[i] = a[i] >= u[i] ? u[i] : (a[i] <= l[i] ? l[i] : a[i])
  end
  return
end
# Routines #####################################################################
