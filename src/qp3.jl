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
  debug::Bool,
)
  #(threadIdx().x != 1 || blockIdx().x != 1) && (return)
  Pnnz, Annz = length(Pi), length(Ai)
  Hnnz = Pnnz + 5 * Annz + m

  ATp, iwork = @view_mem(iwork, m + 1)
  ATi, iwork = @view_mem(iwork, Annz)
  ATx, fwork = @view_mem(fwork, Annz)

  #method = "indirect"

  rho0, sig = 1e1, 1e-12

  #if method == "indirect"
  if true
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

    #if debug
    #  ATA = SparseMatrixCSC(
    #    n,
    #    n,
    #    collect(copy(ATAp)),
    #    collect(copy(view(ATAi, 1:ATAp[n+1] - 1))),
    #    collect(copy(view(ATAx, 1:ATAp[n+1]- 1))),
    #  )
    #  D = ATA - rho0 * (A' * A)
    #  err = norm(D)
    #  @printf("err = %9.4e\n", err)
    #  if err > 1e-12
    #    @infiltrate
    #  end
    #end

    spmatadd!(Tp, Ti, Tx, Pp, Pi, Px, Ip, Ii, Ix)
    spmatadd!(Hp, Hi, Hx, Tp, Ti, Tx, ATAp, ATAi, ATAx)
    sptriu!(Hp, Hi, Hx)

    # CHECK
    #if debug
    #  Hp, Hi, Hx = view(Hp, 1:n+1), view(Hi, 1:Hnnz), view(Hx, 1:Hnnz)
    #  H = SparseMatrixCSC(n, n, collect(Hp), collect(Hi), collect(Hx))
    #  H_ = (P + sig * I + rho0 * A' * A)
    #  err = norm(UpperTriangular(H - H_))
    #  @printf("err = %9.4e\n", err)
    #  if err > 1e-12
    #    @infiltrate
    #  end
    #  @assert err <= 1e-12
    #end

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
      if debug
        rp = 0.0
        for i in 1:n
          rp += (z[i] - zproj[i])^2
        end
        rp = sqrt(rp)

        spmul!(temp_x, Pp, Pi, Px, x)
        spmul!(temp_x2, ATp, ATi, ATx, y)
        rd = 0.0
        for i in 1:n
          rd += (temp_x[i] + q[i] + temp_x2[i])^2
        end
        rd = sqrt(rd)
      end
      #(debug) && (@printf("%9.4e - %9.4e\n", rp, rd))

      # z = zproj
      veccpy!(z, zproj)
    end
    for i in 1:n
      sol[i] = x[i]
    end
    for i in 1:m
      sol[n + i] = y[i]
    end
    return
  end
  #elseif method == "direct"
  #  # TODO; not working yet
  #  rho = [rho0 * (l[i] == u[i] ? 1e3 : 1.0) for i in 1:m]
  #
  #  Hp, iwork = @view_mem(iwork, n + 1)
  #  Hi, iwork = @view_mem(iwork, Hnnz)
  #  Hx, fwork = @view_mem(fwork, Hnnz)

  #  Lnz, iwork = @view_mem(iwork, n + m)
  #  info, iwork = @view_mem(iwork, 1)
  #  etree, iwork = @view_mem(iwork, n + m)
  #  ldlt_iwork, iwork = @view_mem(iwork, 30 * (n + m))
  #  ldlt_fwork, fwork = @view_mem(fwork, 10 * (n + m))

  #  Ip, iwork = @view_mem(iwork, n + m + 1)
  #  Ii, iwork = @view_mem(iwork, n + m)
  #  Ix, fwork = @view_mem(fwork, n + m)
  #  for i in 1:(n+m)
  #    Ip[i], Ii[i], Ix[i] = i, i, (i <= n ? sig : -1.0 / rho[i-n])
  #  end
  #  Ip[(n+m)+1] = (n + m) + 1

  #  Tp, iwork = @view_mem(iwork, n + m + 1)
  #  Ti, iwork = @view_mem(iwork, Hnnz)
  #  Tx, fwork = @view_mem(fwork, Hnnz)

  #  sptranspose!(ATp, ATi, ATx, Ap, Ai, Ax)
  #  spmatadd!(Tp, Ti, Tx, Pp, Pi, Px, Ip, Ii, Ix)
  #  sphcat!(Tp, Ti, Tx, Pp, Pi, Px, ATp, ATi, ATx)
  #  spmatadd!(Hp, Hi, Hx, Tp, Ti, Tx, Ip, Ii, Ix)
  #  sptriu!(Hp, Hi, Hx)

  #  LDLT_etree!(n + m, Hp, Hi, ldlt_iwork, Lnz, info, etree)
  #  @assert info[1] > 0
  #  Lnnz = info[1]

  #  Lp, iwork = @view_mem(iwork, n + m + 1)
  #  Li, iwork = @view_mem(iwork, Lnnz)
  #  D, fwork = @view_mem(fwork, n + m)
  #  Dinv, fwork = @view_mem(fwork, n + m)
  #  Lx, fwork = @view_mem(fwork, Lnnz)

  #  LDLT_factor!(
  #    n + m,
  #    Hp,
  #    Hi,
  #    Hx,
  #    Lp,
  #    Li,
  #    Lx,
  #    D,
  #    Dinv,
  #    info,
  #    Lnz,
  #    etree,
  #    ldlt_iwork,
  #    ldlt_fwork,
  #    false,
  #  )
  #  @assert info[1] > 0


  #  # CHECK
  #  #H = SparseMatrixCSC(n + m, n + m, collect(Hp), collect(Hi), collect(Hx))
  #  #H_ = [
  #  #  P+sig*I A'
  #  #  A -spdiagm(0 => (1.0 ./ rho))
  #  #]
  #  #@infiltrate
  #end

  #x = zeros(n)
  #z = zeros(n + m)
  #y = zeros(n + m)
  #v = zeros(n + m)

  #if method == "indirect"
  #  H = (P + sig * I + rho0 * A' * A)
  #elseif method == "direct"
  #  H = [
  #    P+sig*I A'
  #    A -spdiagm(0 => (1.0 ./ rho))
  #  ]
  #end
  #F = ldlt(H)

  #for i in 1:100
  #  if method == "indirect"
  #    x = F \ (sig * x - q + A' * (rho0 * z - y))
  #    zt = A * x
  #    zclamped = clamp.(zt + y / rho0, l, u)
  #    y = y + rho0 * (zt - zclamped)
  #  elseif method == "direct"
  #    x_ = F \ [
  #      sig * x - q
  #      z - y ./ rho
  #    ]
  #    x, v = x_[1:n], x_[n+1:end]
  #    zt = z + (v - y) ./ rho
  #    zclamped = clamp.(zt + y ./ rho, l, u)
  #    y = y + rho .* (zt - zclamped)
  #  end

  #  rp = norm(A * x - z)
  #  z = zclamped
  #  rd = norm(P * x + q + A' * y)
  #  @printf("%9.4e  %9.4e\n", rp, rd)
  #end
  return
end
