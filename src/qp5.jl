################################################################################
function QP_solve!(
  sol,
  n,
  m,
  Pp,
  Pi,
  Px,
  q,
  Ap,
  Ai,
  Ax,
  l,
  u,
  iwork,
  fwork,
  debug,
)
  #(threadIdx().x != 1 || blockIdx().x != 1) && (return)

  Pnnz, Annz = length(Pi), length(Ai)
  Hnnz = Pnnz + 5 * Annz + m + n

  ATp, iwork = @view_mem(iwork, m + 1)
  ATi, iwork = @view_mem(iwork, Annz)
  ATx, fwork = @view_mem(fwork, Annz)

  rho0, sig = 1f1, 1f-12

  Hp, iwork = @view_mem(iwork, n + m + 1)
  Hi, iwork = @view_mem(iwork, Hnnz)
  Hx, fwork = @view_mem(fwork, Hnnz)

  Lnz, iwork = @view_mem(iwork, n + m)
  info, iwork = @view_mem(iwork, 1)
  etree, iwork = @view_mem(iwork, n + m)
  ldlt_iwork, iwork = @view_mem(iwork, 30 * (n + m))
  ldlt_fwork, fwork = @view_mem(fwork, 10 * (n + m))

  Ip, iwork = @view_mem(iwork, n + m + 1)
  Ii, iwork = @view_mem(iwork, n + m)
  Ix, fwork = @view_mem(fwork, n + m)
  for i in 1:n
    Ip[i], Ii[i], Ix[i] = i, i, sig
  end
  for i in 1:m
    Ip[n + i], Ii[n + i], Ix[n + i] = i + n, i + n, -1f0 / rho0
  end
  Ip[n+m+1] = n + m + 1

  Tp, iwork = @view_mem(iwork, n + m + 1)
  Ti, iwork = @view_mem(iwork, Hnnz)
  Tx, fwork = @view_mem(fwork, Hnnz)

  sptranspose!(ATp, ATi, ATx, Ap, Ai, Ax)
  sphcat!(Tp, Ti, Tx, Pp, Pi, Px, ATp, ATi, ATx)
  Ti = view(Ti, 1:Tp[n + m + 1]-1)
  Tx = view(Tx, 1:Tp[n + m + 1]-1)
  spmatadd!(Hp, Hi, Hx, Tp, Ti, Tx, Ip, Ii, Ix)
  sptriu!(Hp, Hi, Hx)

  LDLT_etree!(n + m, Hp, Hi, ldlt_iwork, Lnz, info, etree)
  @assert info[1] > 0
  Lnnz = info[1]

  Lp, iwork = @view_mem(iwork, n + m + 1)
  Li, iwork = @view_mem(iwork, Lnnz)
  D, fwork = @view_mem(fwork, n + m)
  Dinv, fwork = @view_mem(fwork, n + m)
  Lx, fwork = @view_mem(fwork, Lnnz)

  LDLT_factor!(
    n + m,
    Hp,
    Hi,
    Hx,
    Lp,
    Li,
    Lx,
    D,
    Dinv,
    info,
    Lnz,
    etree,
    ldlt_iwork,
    ldlt_fwork,
    false,
  )
  @assert info[1] > 0

  temp, fwork = @view_mem(fwork, n + m)
  x, fwork = @view_mem(fwork, n)
  z, fwork = @view_mem(fwork, m)
  zproj, fwork = @view_mem(fwork, m)
  y, fwork = @view_mem(fwork, m)
  v, fwork = @view_mem(fwork, m)
  temp_x, fwork = @view_mem(fwork, n)
  temp_x2, fwork = @view_mem(fwork, n)

  for i in 1:50
    # x = [sig * x - q z - y ./ rho0]
    for i in 1:n
      temp[i] = sig * x[i] - q[i]
    end
    for i in 1:m
      temp[n + i] = z[i] - y[i] / rho0
    end

    # LDLT_solve!(n + m, Lp, Li, Lx, Dinv, x)
    LDLT_solve!(n + m, Lp, Li, Lx, Dinv, temp)
    
    # x, v = x[1:n], x[n+1:end]
    veccpy!(x, view(temp, 1:n))
    veccpy!(v, view(temp, n+1:n+m))

    # z = z + (v - y) ./ rho0
    for i in 1:m
      z[i] += (v[i] - y[i]) / rho0
    end
    
    # zproj = clamp.(z + y ./ rho0, l, u)
    for i in 1:m
      zproj[i] = z[i] + y[i] / rho0
    end
    vecclamp!(zproj, zproj, l, u)

    # y = y + rho .* (z - zclamped)
    for i in 1:m
      y[i] += rho0 * (z[i] - zproj[i])
    end

    # compute residuals
    # if debug
    #   rp = 0f0
    #   for i in 1:n
    #     rp += (z[i] - zproj[i])^2
    #   end
    #   rp = sqrt(rp)

    #   #spmul!(temp_x, Pp, Pi, Px, x)
    #   #spmul!(temp_x2, ATp, ATi, ATx, y)
    #   rd = 0f0
    #   for i in 1:n
    #     rd += (temp_x[i] + q[i] + temp_x2[i])^2
    #   end
    #   rd = sqrt(rd)
    # end
    # (debug) && (@printf("%9.4e - %9.4e\n", rp, rd))

    # z = zproj
    veccpy!(z, zproj)
  end

  veccpy!(view(sol, 1:n), x)
  veccpy!(view(sol, n+1:n+m), y)

  return
end
