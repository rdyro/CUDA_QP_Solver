################################################################################
function QP_solve!(sol, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, b, l, u, iwork, fwork)
  #=
  if threadIdx().x != 1 || blockIdx().x != 1
  return
  end
  =#
  Pnnz, Annz = length(Pi), length(Ai)
  Hnnz = Pnnz + 5 * Annz + m

  is, ie = 0, 0
  is, ie = ie + 1, ie + n + m + 1
  Hp = view(iwork, is:ie)
  is, ie = ie + 1, ie + Hnnz
  Hi = view(iwork, is:ie)
  is, ie = ie + 1, ie + m + 1
  ATp = view(iwork, is:ie)
  is, ie = ie + 1, ie + Annz + m
  ATi = view(iwork, is:ie)
  is, ie = ie + 1, ie + n + 1
  ATAp = view(iwork, is:ie)
  is, ie = ie + 1, ie + 5 * Annz
  ATAi = view(iwork, is:ie)
  is, ie = ie + 1, ie + n + m
  Lnz = view(iwork, is:ie)
  is, ie = ie + 1, ie + 1
  info = view(iwork, is:ie)
  is, ie = ie + 1, ie + n + m
  etree = view(iwork, is:ie)
  #is, ie = ie + 1, ie + 4 * (n + m);      ldlt_iwork = view(iwork, is:ie)
  is, ie = ie + 1, ie + 30 * (n + m)
  ldlt_iwork = view(iwork, is:ie)
  #ilen = 2 * Annz + Pnnz + 10 * m + 7 * n + 3

  fs, fe = 0, 0
  fs, fe = fe + 1, fe + Hnnz
  Hx = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + Annz + m
  ATx = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + 5 * Annz
  ATAx = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n
  x = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n
  xp = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n
  y = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n
  yp = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n
  z = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n
  zp = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n
  v = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n + m
  ldlt_fwork = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + 10 * (n + m)
  ldlt_fwork = view(fwork, fs:fe)
  #flen = 2 * Annz + Pnnz + 4 * m  + 3 * n

  rho = 1e1
  #=
  # upper triangular cat
  #print("Transpose = ")
  #@time sptranspose!(ATp, ATi, ATx, Ap, Ai, Ax)
  sptranspose!(ATp, ATi, ATx, Ap, Ai, Ax)

  #print("Quad prod = ")
  #@time spmatmulATB!(ATAp, ATAi, ATAx, Ap, Ai, Ax, m, n, Ap, Ai, Ax, m, n)
  #@time spmatmul!(ATAp, ATAi, ATAx, n, ATp, ATi, ATx, Ap, Ai, Ax, ldlt_iwork)
  spmatmul!(ATAp, ATAi, ATAx, n, ATp, ATi, ATx, Ap, Ai, Ax, ldlt_iwork)
  ATAp, ATAi = view(ATAp, 1:n+1), view(ATAi, 1:ATAp[n+1]-1)
  ATAx = view(ATAx, 1:ATAp[n+1]-1)

  spscal!(ATAp, ATAi, ATAx, rho)
  #print("Adding = ")
  #@time spmatadd!(Hp, Hi, Hx, Pp, Pi, Px, ATAp, ATAi, ATAx)
  spmatadd!(Hp, Hi, Hx, Pp, Pi, Px, ATAp, ATAi, ATAx)
  sptriu!(Hp, Hi, Hx)
  Hnnz = Hp[n+1]
  Hp, Hi, Hx = view(Hp, 1:n+1), view(Hi, 1:Hnnz), view(Hx, 1:Hnnz)

  #print("Tree = ")
  #@time LDLT_etree!(n, Hp, Hi, ldlt_iwork, Lnz, info, etree)
  LDLT_etree!(n, Hp, Hi, ldlt_iwork, Lnz, info, etree)
  @assert info[1] > 0
  Lnnz = info[1]

  is, ie = ie + 1, ie + n + m + 1;        Lp = view(iwork, is:ie)
  is, ie = ie + 1, ie + Lnnz;             Li = view(iwork, is:ie)
  fs, fe = fe + 1, fe + n + m;            D = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n + m;            Dinv = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + Lnnz;             Lx = view(fwork, fs:fe)

  #print("Factor = ")
  #@time LDLT_factor!(n, Hp, Hi, Hx, Lp, Li, Lx, D, Dinv, info, Lnz, etree,
  #                   ldlt_iwork, ldlt_fwork, false)
  LDLT_factor!(n, Hp, Hi, Hx, Lp, Li, Lx, D, Dinv, info, Lnz, etree,
  ldlt_iwork, ldlt_fwork, false)
  @assert info[1] > 0

  z = view(z, 1:m)
  y = view(y, 1:m)
  v = view(v, 1:m)

  vecfill!(z, 0.0)
  vecfill!(y, 0.0)
  res = view(ldlt_fwork, 1:n)
  sol = view(sol, 1:n)

  #print("Solve = ")
  #@time for k in 1:25
  for k in 1:25
  vecfill!(sol, 0.0)
  veccpy!(v, z)
  vecsub!(v, y)
  #@btime spmul!($sol, $ATp, $ATi, $ATx, $v)
  spmul!(sol, ATp, ATi, ATx, v)
  vecscal!(sol, rho)
  vecsub!(sol, q)

  #@btime LDLT_solve!($n, $Lp, $Li, $Lx, $Dinv, $sol)
  LDLT_solve!(n, Lp, Li, Lx, Dinv, sol)

  #@btime spmul!($v, $Ap, $Ai, $Ax, $sol)
  spmul!(v, Ap, Ai, Ax, sol)
  veccpy!(z, v)
  vecadd!(z, y)
  vecclamp!(z, z, b, b)
  vecadd!(y, v)
  vecsub!(y, z)
  end

  return
  =#

  sig = 1e-7
  Ip, Ii = view(ldlt_iwork, 1:m+1), view(ldlt_iwork, m+1+1:m+1+m)
  Ix = view(ldlt_fwork, 1:m)
  Ip[1] = 1
  for i in 1:m  #= @inbounds =#
    Ip[i+1] = i + 1
    Ii[i] = i
    Ix[i] = sig
  end
  sphcat!(Hp, Hi, Hx, Ap, Ai, Ax, Ip, Ii, Ix)
  sptranspose!(
    ATp,
    ATi,
    ATx,
    view(Hp, 1:n+m+1),
    view(Hi, 1:Annz+m),
    view(Hx, 1:Annz+m),
  )

  spcopy!(Hp, Hi, Hx, Pp, Pi, Px)
  Hnnz = Hp[n+1] - 1
  sptriu!(Hp, Hi, Hx)
  sphcat!(
    Hp,
    Hi,
    Hx,
    view(Hp, 1:n+1),
    view(Hi, 1:Hnnz),
    view(Hx, 1:Hnnz),
    ATp,
    ATi,
    ATx,
  )
  Hnnz = Hp[n+m+1] - 1
  spdiagadd!(Hp, Hi, Hx, 1, n, sig)
  spdiagadd!(Hp, Hi, Hx, 1, n, rho)

  LDLT_etree!(n + m, Hp, Hi, ldlt_iwork, Lnz, info, etree)
  @assert info[1] > 0
  Lnnz = info[1]

  is, ie = ie + 1, ie + n + m + 1
  Lp = view(iwork, is:ie)
  is, ie = ie + 1, ie + Lnnz
  Li = view(iwork, is:ie)
  fs, fe = fe + 1, fe + n + m
  D = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + n + m
  Dinv = view(fwork, fs:fe)
  fs, fe = fe + 1, fe + Lnnz
  Lx = view(fwork, fs:fe)

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
  for k in 1:50
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

    #veccpy!(res, view(sol, 1:n))
    #vecsub!(res, z)
    #norm = vecdot(res, res)
    #println(norm)

    #veccpy!(res, view(sol, 1:n))
    #vecsub!(res, z)
    #norm = vecdot(res, res)
    #println(norm)
  end
  return
end
