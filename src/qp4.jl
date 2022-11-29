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
  Hnnz = Pnnz + 5 * Annz + m

  ATp, iwork = @view_mem(iwork, m + 1)
  ATi, iwork = @view_mem(iwork, Annz)
  ATx, fwork = @view_mem(fwork, Annz)

  rho0, sig = 1f1, 1f-12

  Hp, iwork = @view_mem(iwork, n + 1)
  Hi, iwork = @view_mem(iwork, Hnnz)
  Hx, fwork = @view_mem(fwork, Hnnz)

  Lnz, iwork = @view_mem(iwork, n)
  info, iwork = @view_mem(iwork, 1)
  etree, iwork = @view_mem(iwork, n)
  ldlt_iwork, iwork = @view_mem(iwork, 30 * n)
  ldlt_fwork, fwork = @view_mem(fwork, 10 * n)

  ATAp, iwork = @view_mem(iwork, n + 1)
  ATAi, iwork = @view_mem(iwork, 3 * Annz)
  ATAx, fwork = @view_mem(fwork, 3 * Annz)

  Ip, iwork = @view_mem(iwork, n + 1)
  Ii, iwork = @view_mem(iwork, n)
  Ix, fwork = @view_mem(fwork, n)
  for i in 1:n
    Ip[i], Ii[i], Ix[i] = i, i, sig
  end
  Ip[n+1] = n + 1

  Tp, iwork = @view_mem(iwork, n + 1)
  Ti, iwork = @view_mem(iwork, Hnnz)
  Tx, fwork = @view_mem(fwork, Hnnz)

  sptranspose!(ATp, ATi, ATx, Ap, Ai, Ax)
  spmatmul!(ATAp, ATAi, ATAx, n, ATp, ATi, ATx, Ap, Ai, Ax, iwork)
  vecscal!(ATAx, rho0)

  spmatadd!(Tp, Ti, Tx, Pp, Pi, Px, Ip, Ii, Ix)
  spmatadd!(Hp, Hi, Hx, Tp, Ti, Tx, ATAp, ATAi, ATAx)
  sptriu!(Hp, Hi, Hx)

  LDLT_etree!(n, Hp, Hi, ldlt_iwork, Lnz, info, etree)
  @assert info[1] > 0
  Lnnz = info[1]

  Lp, iwork = @view_mem(iwork, n + 1)
  Li, iwork = @view_mem(iwork, Lnnz)
  D, fwork = @view_mem(fwork, n)
  Dinv, fwork = @view_mem(fwork, n)
  Lx, fwork = @view_mem(fwork, Lnnz)

  LDLT_factor!(
    n,
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

  x, fwork = @view_mem(fwork, n)
  z, fwork = @view_mem(fwork, m)
  zproj, fwork = @view_mem(fwork, m)
  y, fwork = @view_mem(fwork, m)
  temp_x, fwork = @view_mem(fwork, n)
  temp_x2, fwork = @view_mem(fwork, n)
  temp_z, fwork = @view_mem(fwork, m)

  for i in 1:100
    # x = (sig * x - q + A' * (rho0 * z - y))
    for i in 1:n
      temp_z[i] = rho0 * z[i] - y[i]
    end
    spmul!(temp_x, ATp, ATi, ATx, temp_z)
    for i in 1:n
      x[i] = sig * x[i] - q[i] + temp_x[i]
    end

    # x = F \ x
    LDLT_solve!(n, Lp, Li, Lx, Dinv, x)

    # z = A * x
    spmul!(z, Ap, Ai, Ax, x)

    # zproj = clamp.(z + y / rho0, l, u)
    for i in 1:m
      zproj[i] = z[i] + y[i] / rho0
    end
    vecclamp!(zproj, zproj, l, u)

    # y = y + rho0 * (z - zproj)
    for i in 1:m
      y[i] += rho0 * (z[i] - zproj[i])
    end

    # compute residuals
    #if debug
    #  rp = 0f0
    #  for i in 1:n
    #    rp += (z[i] - zproj[i])^2
    #  end
    #  rp = sqrt(rp)

    #  #spmul!(temp_x, Pp, Pi, Px, x)
    #  #spmul!(temp_x2, ATp, ATi, ATx, y)
    #  rd = 0f0
    #  for i in 1:n
    #    rd += (temp_x[i] + q[i] + temp_x2[i])^2
    #  end
    #  rd = sqrt(rd)
    #end
    #(debug) && (@printf("%9.4e - %9.4e\n", rp, rd))

    # z = zproj
    veccpy!(z, zproj)
  end
  
  veccpy!(view(sol, 1:n), x)
  veccpy!(view(sol, n + 1:n+m), y)
  
  return
end
