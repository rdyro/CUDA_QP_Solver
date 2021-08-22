function qsort!(a, lo, hi)
  if hi - lo >= 1
    p = pivot!(a, lo, hi)
    qsort!(a, lo, p)
    qsort!(a, p+1, hi)
  end
end

function pivot!(a, lo, hi)
  if hi - lo <= 1
    return lo
  elseif hi - lo == 2
    a[lo], a[hi] = a[hi] > a[lo] ? (a[lo], a[hi]) : (a[hi], a[lo])
    return lo
  end
  pval = a[div(hi + lo, 2)]
  i, j = lo - 1, hi + 1
  while true
    i += 1
    while a[i] < pval
        i = i + 1
    end
    j -= 1
    while a[j] > pval
        j = j - 1
    end
    if i >= j
      return j
    end
    a[i], a[j] = a[j], a[i]
  end
end

function transpose!(Cp, Ci, Cx, Ap, Ai, Ax, iwork)
  # needs 4 x length(Ai) in iwork
  nnz, n, m = length(Ai), length(Ap) - 1, length(Cp) - 1
  #@assert nnz < (1 << 21) && n < (1 << 21) && m < (2 << 21)
  I = view(iwork, 1:nnz)
  J = view(iwork, nnz+1:2*nnz)
  pos = view(iwork, 2*nnz+1:3*nnz)
  pos_iwork = view(iwork, 3*nnz+1:4*nnz)
  k = 0
  for i in 1:n
    for j in Ap[i]:Ap[i+1]-1
      k += 1
      I[k] = Ai[j]
      J[k] = i
      pos[k] = (I[k] << 42) | (J[k] << 21) | k
    end
  end
  @time sort!(pos)
  @time mergesort!(pos, 1, nnz, pos_iwork)
  #insertionsort!(pos, 1, nnz)
  Cp[1] = 1
  k = 1
  idx = pos[k] & 0x1fffff
  for i in 1:m
    k_old = k
    while k <= nnz && I[idx] == i
      Ci[k] = J[idx]
      Cx[k] = Ax[idx]
      k += 1
      idx = k <= nnz ? pos[k] & 0x1fffff : -1
    end
    Cp[i+1] = Cp[i] + (k - k_old)
  end
  return
end

@inline function binary_search(a, i)
  if a[end] < i || a[1] > i
    return -1
  end
  s, e = 1, length(a)
  while e - s > 1
    m = div(e + s, 2)
    if a[m] == i
      return m
    elseif a[m] > i
      e = m
    else
      s = m
    end
  end
  return a[e] == i ? e : (a[s] == i ? s : -1)
end

function insertionsort!(a, lo, hi)
  for i in lo:hi
    min_val, min_idx = Inf, -1
    for j in i+1:hi
      if a[j] < min_val
        min_val = a[j]
        min_idx = j
      end
    end
    if min_val < a[i]
      a[i], a[min_idx] = a[min_idx], a[i]
    end
  end
  return
end

