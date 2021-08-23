################################################################################
function QP_solve!(sol, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, b, l, u, iwork, fwork)
  (threadIdx().x != 1 || blockIdx().x != 1) && (return)
  Pnnz, Annz = length(Pi), length(Ai)
  Hnnz = Pnnz + 5 * Annz + m

  Hp, iwork = @view_mem(iwork, n + m + 1)
  Hi, iwork = @view_mem(iwork, Hnnz)
  ATp, iwork = @view_mem(iwork, m + 1)
  ATi, iwork = @view_mem(iwork, Annz + m)
  ATAp, iwork = @view_mem(iwork, n + 1)
  ATAi, iwork = @view_mem(iwork, 5 * Annz)
  Lnz, iwork = @view_mem(iwork, n + m)
  info, iwork = @view_mem(iwork, 1)
  etree, iwork = @view_mem(iwork, n + m)
  ldlt_iwork, iwork = @view_mem(iwork, 30 * (n + m))

  Hx, fwork = @view_mem(fwork, Hnnz)
  ATx, fwork = @view_mem(fwork, Annz + m)
  ATAx, fwork = @view_mem(fwork, 5 * Annz)
  x, fwork = @view_mem(fwork, n)
  xp, fwork = @view_mem(fwork, n)
  y, fwork = @view_mem(fwork, n)
  yp, fwork = @view_mem(fwork, n)
  z, fwork = @view_mem(fwork, n)
  zp, fwork = @view_mem(fwork, n)
  v, fwork = @view_mem(fwork, n)
  ldlt_fwork, fwork = @view_mem(fwork, 10 * (n + m))

  rho, sig = 1e1, 1e-7
  Ip, Ii = view(ldlt_iwork, 1:m+1), view(ldlt_iwork, m+1+1:m+1+m)
  Ix = view(ldlt_fwork, 1:m)
  Ip[1] = 1
  for i in 1:m
    Ip[i+1] = i + 1
    Ii[i] = i
    Ix[i] = sig
  end
  sphcat!(Hp, Hi, Hx, Ap, Ai, Ax, Ip, Ii, Ix)
  Hp_, Hi_, Hx_ = view(Hp, 1:n+m+1), view(Hi, 1:Annz+m), view(Hx, 1:Annz+m)
  sptranspose!(ATp, ATi, ATx, Hp_, Hi_, Hx_)

  spcopy!(Hp, Hi, Hx, Pp, Pi, Px)
  Hnnz = Hp[n+1] - 1
  sptriu!(Hp, Hi, Hx)
  Hp_, Hi_, Hx_ = view(Hp, 1:n+1), view(Hi, 1:Hnnz), view(Hx, 1:Hnnz)
  sphcat!(Hp, Hi, Hx, Hp_, Hi_, Hx_, ATp, ATi, ATx)
  Hnnz = Hp[n+m+1] - 1
  spdiagadd!(Hp, Hi, Hx, 1, n, sig)
  spdiagadd!(Hp, Hi, Hx, 1, n, rho)

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

  vecfill!(z, 0.0)
  vecfill!(zp, 0.0)
  vecfill!(y, 0.0)
  vecfill!(yp, 0.0)
  vecfill!(x, 0.0)
  vecfill!(xp, 0.0)
  res = view(ldlt_fwork, 1:n)
  for k in 1:100
    vecfill!(sol, 0.0)
    vecsub!(sol, y)
    vecadd!(sol, z)
    vecscal!(sol, rho)
    vecsub!(sol, q)
    veccpy!(view(sol, n+1:n+m), b)

    LDLT_solve!(n + m, Lp, Li, Lx, Dinv, sol)
    veccpy!(z, view(sol, 1:n))
    vecadd!(z, y)

    vecclamp!(z, z, l, u)
    vecadd!(y, view(sol, 1:n))
    vecsub!(y, z)

    veccpy!(res, view(sol, 1:n))
    vecsub!(res, z)
    norm = vecdot(res, res)
    println(norm)
  end
  return
end
