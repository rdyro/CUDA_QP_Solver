# Utils ########################################################################
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

function merge!(a, lo, mi, hi, iwork)
  i, j, k = lo, mi + 1, lo
  while i <= mi && j <= hi  #= @inbounds =#
    if a[i] > a[j]
      iwork[k] = a[j]
      j += 1
      k += 1
    else
      iwork[k] = a[i]
      i += 1
      k += 1
    end
  end
  while i <= mi  #= @inbounds =#
    iwork[k] = a[i]
    i += 1
    k += 1
  end
  while j <= hi  #= @inbounds =#
    iwork[k] = a[j]
    j += 1
    k += 1
  end
  return
end

function mergesort!(a, lo, hi, iwork)
  nb = hi - lo
  (nb < 1) && (return)
  if nb == 2
    if a[lo] > a[hi]
      a[lo], a[hi] = a[hi], a[lo]
    end
  end

  step, odd_run = 1, true
  ##= @inbounds =# while step < nb || !odd_run
  while step <= nb  #= @inbounds =#
    for i in 1:div(nb, 2 * step)+1    #= @inbounds =#
      loi = min(lo + 2 * (i - 1) * step, hi)
      mii = min(lo + 2 * (i - 1) * step + step - 1, hi)
      hii = min(lo + 2 * (i - 1) * step + 2 * step - 1, hi)
      #println("loi = ", loi, " mii = ", mii, " hii = ", hii)
      if odd_run
        merge!(a, loi, mii, hii, iwork)
      else
        merge!(iwork, loi, mii, hii, a)
      end
    end
    odd_run = !odd_run
    step *= 2
  end
  if !odd_run
    for i in lo:hi
      a[i] = iwork[i]
    end
  end
  return
end
# Utils ########################################################################

# Routines #####################################################################
function sptriu!(Ap, Ai, Ax)
  k, k_old, n = 0, 0, length(Ap) - 1
  Aps, Ape = Ap[1], Ap[2] - 1
  for i in 1:n  #= @inbounds =#
    for j in Aps:Ape    #= @inbounds =#
      row, col = Ai[j], i
      if row <= col
        k += 1
        Ai[k] = row
        Ax[k] = Ax[j]
      end
    end
    if i < n
      Aps, Ape = Ap[i+1], Ap[i+2] - 1
    end
    Ap[i+1] = Ap[i] + (k - k_old)
    k_old = k
  end
  return
end

function sptril!(Ap, Ai, Ax)
  k, k_old, n = 0, 0, length(Ap) - 1
  Aps, Ape = Ap[1], Ap[2] - 1
  for i in 1:n  #= @inbounds =#
    for j in Aps:Ape    #= @inbounds =#
      row, col = Ai[j], i
      if row >= col
        k += 1
        Ai[k] = row
        Ax[k] = Ax[j]
      end
    end
    if i < n
      Aps, Ape = Ap[i+1], Ap[i+2] - 1
    end
    Ap[i+1] = Ap[i] + (k - k_old)
    k_old = k
  end
  return
end

function spscal!(Ap, Ai, Ax, alf)
  n = length(Ap) - 1
  for i in 1:n  #= @inbounds =#
    for j in Ap[i]:Ap[i+1]-1    #= @inbounds =#
      Ax[j] = Ax[j] * alf
    end
  end
end

function spmul!(y, Ap, Ai, Ax, x) # GPU ok
  n = length(Ap) - 1
  vecfill!(y, 0.0f0)
  for j in 1:n  #= @inbounds =#
    for i in Ap[j]:Ap[j+1]-1    #= @inbounds =#
      y[Ai[i]] += Ax[i] * x[j]
    end
  end
end

function spmatadd!(Cp, Ci, Cx, Ap, Ai, Ax, Bp, Bi, Bx) # GPU ok
  Cp[1] = 1
  l, l_old = 0, 0
  n = length(Ap) - 1
  for k in 1:n  #= @inbounds =#
    An, Bn = Ap[k+1] - Ap[k], Bp[k+1] - Bp[k]
    As, Bs = Ap[k] - 1, Bp[k] - 1
    i, j = 1, 1
    while i <= An || j <= Bn    #= @inbounds =#
      if i <= An && j <= Bn && Ai[As+i] == Bi[Bs+j]
        l += 1
        Ci[l] = Ai[As+i]
        Cx[l] = Ax[As+i] + Bx[Bs+j]

        i, j = i + 1, j + 1
      elseif i > An || (j <= Bn && Bi[Bs+j] < Ai[As+i])
        l += 1
        Ci[l] = Bi[Bs+j]
        Cx[l] = Bx[Bs+j]

        j += 1
      else
        l += 1
        Ci[l] = Ai[As+i]
        Cx[l] = Ax[As+i]

        i += 1
      end
    end
    Cp[k+1] = Cp[k] + (l - l_old)
    l_old = l
  end
  return Cp, view(Ci, 1:l), view(Cx, 1:l)
end

@inline function spvecdot(Ai, Ax, Bi, Bx, check_only=false)
  x, iszero = 0.0f0, true
  i, j = 1, 1
  n, m = length(Ai), length(Bi)
  while i <= n && j <= m  #= @inbounds =#
    while i <= n && j <= m && Ai[i] < Bi[j]
      i += 1
    end
    while i <= n && j <= m && Bi[j] < Ai[i]
      j += 1
    end
    if i <= n && j <= m && Ai[i] == Bi[j]
      if check_only
        return 1.0f0, false
      end
      iszero = false
      x += Ax[i] * Bx[j]
      i += 1
      j += 1
    end
  end
  return x, iszero
end


function spcopy!(Cp, Ci, Cx, Ap, Ai, Ax)
  for i in 1:length(Ap)  #= @inbounds =#
    Cp[i] = Ap[i]
  end
  for i in 1:length(Ai)  #= @inbounds =#
    Ci[i] = Ai[i]
    Cx[i] = Ax[i]
  end
end

function spvcat!(Cp, Ci, Cx, m, Ap, Ai, Ax, Bp, Bi, Bx)
  l = 0
  Cp[1] = 1
  for i in 1:length(Ap)-1  #= @inbounds =#
    for j in Ap[i]:Ap[i+1]-1    #= @inbounds =#
      l += 1
      Ci[l] = Ai[j]
      Cx[l] = Ax[j]
    end
    for j in Bp[i]:Bp[i+1]-1    #= @inbounds =#
      l += 1
      Ci[l] = Bi[j] + m
      Cx[l] = Bx[j]
    end
    Cp[i+1] = Cp[i] + (Ap[i+1] - Ap[i]) + (Bp[i+1] - Bp[i])
  end
  return
end

function sphcat!(Cp, Ci, Cx, Ap, Ai, Ax, Bp, Bi, Bx)
  n1, n2 = length(Ap) - 1, length(Bp) - 1
  for i in 1:n2  #= @inbounds =#
    Cp[n1+i+1] = Bp[i+1] + Ap[end] - 1
  end
  for i in 1:n1+1  #= @inbounds =#
    Cp[i] = Ap[i]
  end
  Annz, Bnnz = length(Ax), length(Bx)
  for i in 1:Bnnz  #= @inbounds =#
    Ci[i+Annz] = Bi[i]
    Cx[i+Annz] = Bx[i]
  end
  for i in 1:Annz  #= @inbounds =#
    Ci[i] = Ai[i]
    Cx[i] = Ax[i]
  end
  return
end

# taken from Julia's stdlib/SparseArrays
function sptranspose!(
  Xp::ArrayView{T1},
  Xi::ArrayView{T1},
  Xx::ArrayView{T2},
  Ap::ArrayView{T1},
  Ai::ArrayView{T1},
  Ax::ArrayView{T2},
) where {T1,T2}
  n, nnz = length(Xp) - 1, length(Ai)
  for i in 1:n+1
    Xp[i] = 0
  end
  Xp[1] = 1
  for k in 1:nnz
    Xp[Ai[k]+1] += 1
  end
  countsum = 1
  for k in 2:n+1
    overwritten = Xp[k]
    Xp[k] = countsum
    countsum += overwritten
  end
  for i in 1:(length(Ap)-1)
    for k in (Ap[i]):(Ap[i+1]-1)
      Xk = Xp[Ai[k]+1]
      Xi[Xk] = i
      Xx[Xk] = Ax[k]
      Xp[Ai[k]+1] += 1
    end
  end
  return
end

function spdiagadd!(Cp, Ci, Cx, lo, hi, alf)
  for i in lo:hi  #= @inbounds =#
    col = view(Ci, Cp[i]:(Cp[i+1]-1))
    idx = binary_search(col, i)
    if idx >= 1 && idx <= length(col)
      Cx[Cp[i]+idx-1] = Cx[Cp[i]+idx-1] + alf
    end
  end
  return
end
# Routines #####################################################################

# Mat Mul ######################################################################
prefer_sort(nz, m) = m > 6 && 3 * ceil(log2(nz)) * nz < m

function estimate_mulsize(m, nnzA, n, nnzB, k)
  p = (nnzA / (m * n)) * (nnzB / (n * k))
  p >= 1 ? m * k : p > 0 ? Int(ceil(-expm1(log1p(-p) * n) * m * k)) : 0 # (1-(1-p)^n)*m*k
end

@inline nzrange_(Ap::Array, i) = (Ap[i]):(Ap[i+1]-1)

function spmatmul!(Cp, Ci, Cx, mA, Ap, Ai, Ax, Bp, Bi, Bx, iwork)
  nA, nB = length(Ap) - 1, length(Bp) - 1
  nnzA, nnzB = Ap[end] - 1, Bp[end] - 1
  nnzC = length(Cx)

  #nnzC = min(estimate_mulsize(mA, nnzA, nA, nnzB, nB) * 11 / 10 + mA, mA * nB)
  #nnzC = nnzC * 10

  xb, sort_iwork = view(iwork, 1:mA), view(iwork, mA+1:length(iwork))
  for i in 1:length(xb)
    xb[i] = 0
  end

  ip = 1
  for i in 1:nB
    @cuassert ip + mA - 1 <= nnzC
    Cp[i] = ip
    ip = spcolmul!(Ci, Cx, xb, i, ip, mA, Ap, Ai, Ax, Bp, Bi, Bx, sort_iwork)
  end
  Cp[nB+1] = ip
end

# process single rhs column
@inline function spcolmul!(Ci, Cx, xb, i, ip, mA, Ap, Ai, Ax, Bp, Bi, Bx, sort_iwork)
  ip0 = ip
  k0 = ip - 1
  for jp in (Bp[i]):(Bp[i+1]-1)
    nzB = Bx[jp]
    j = Bi[jp]
    for kp in (Ap[j]):(Ap[j+1]-1)
      nzC = Ax[kp] * nzB
      k = Ai[kp]
      if xb[k] == 1
        Cx[k+k0] += nzC
      else
        Cx[k+k0] = nzC
        xb[k] = 1
        Ci[ip] = k
        ip += 1
      end
    end
  end
  if ip > ip0
    if prefer_sort(ip - k0, mA)
      # in-place sort of indices. Effort: O(nnz*ln(nnz)).
      #sort!(Ci, ip0, ip - 1, QuickSort, Base.Order.Forward)
      mergesort!(Ci, ip0, ip - 1, sort_iwork)
      for vp in ip0:ip-1
        k = Ci[vp]
        xb[k] = 0
        Cx[vp] = Cx[k+k0]
      end
    else
      # scan result vector (effort O(mA))
      for k in 1:mA
        if xb[k] == 1
          xb[k] = 0
          Ci[ip0] = k
          Cx[ip0] = Cx[k+k0]
          ip0 += 1
        end
      end
    end
  end
  return ip
end
# Mat Mul ######################################################################
