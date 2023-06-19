using CUDA, Infiltrator

function test_kernel(data, n)
  #(threadIdx().x != 1 || blockIdx().x != 1) && (return)

  iwork = CUDA.@cuStaticSharedMem(Int32, 10000)
  fwork = CUDA.@cuStaticSharedMem(Float32, 10000)
  a, fwork = view_mem(fwork, len32(data))
  b, fwork = view_mem(fwork, len32(data))
  c, fwork = view_mem(fwork, len32(data))
  d, fwork = view_mem(fwork, len32(data))
  e, fwork = view_mem(fwork, n)

  for i in 1:len32(data)
    #a[i] = data[i]
    b[i] = 1.0 * i
    c[i] = 1.0 * i
    d[i] = 1.0 * 2
  end
  for i in 1:len32(data)
    #data[i] = a[i] * b[i]
    data[i] = b[i] * b[i] - c[i]
  end
  data[end] = len32(fwork)
  return nothing
end

const S = 1000

@inline function has_empty(Ap)
  any_same = false
  for i in 1:(len32(Ap) - Int32(1))
    any_same |= (Ap[i] == Ap[i+1])
  end
  return any_same
end

####################################################################################################


#function QP_solve!(sol, n, m, Pp, Pi, Px, q, Ap, Ai, Ax, l, u)
function QP_solve!(sol, n, m, P, q, A, l, u)
  #(threadIdx().x != 1 || blockIdx().x != 1) && (return)

  a = CuDynamicSharedArray(Int32, 10)
  b = view(a, 1:Int32(5))

  iwork = make_mem(CuDynamicSharedArray(Int32, 2^12))
  fwork = make_mem(CuDynamicSharedArray(Float32, 2^12, 4 * length(iwork)))

  Pp, Pi, Px = P
  Ap, Ai, Ax = A

  # allocate working memory for temporary matrices #############################
  Pnnz, Annz = len32(Pi), len32(Ai)
  Hnnz = Int32(Pnnz + 5 * Annz + m + n)

  ##################################################################################################

  ATi = alloc_mem!(iwork, Annz)
  ATx = alloc_mem!(fwork, Annz)
  ATp = alloc_mem!(iwork, m + 1)

  rho0, sig = 1.0f1, 1.0f-12

  Hi = alloc_mem!(iwork, Hnnz)
  Hx = alloc_mem!(fwork, Hnnz)
  Hp = alloc_mem!(iwork, n + m + 1)

  Lnz = alloc_mem!(iwork, n + m)
  info = alloc_mem!(iwork, 1)
  etree = alloc_mem!(iwork, n + m)
  #ldlt_iwork = alloc_mem!(iwork, 30 * (n + m))
  #ldlt_fwork = alloc_mem!(fwork, 10 * (n + m))
  ldlt_iwork = alloc_mem!(iwork, 4 * (n + m))
  ldlt_fwork = alloc_mem!(fwork, 2 * (n + m))

  # create an identity matrix ##################################################
  Ip = alloc_mem!(iwork, n + m + 1)
  Ii = alloc_mem!(iwork, n + m)
  Ix = alloc_mem!(fwork, n + m)

  for i in 1:n
    Ip[i], Ii[i], Ix[i] = i, i, sig
  end
  for i in 1:m
    Ip[n+i], Ii[n+i], Ix[n+i] = i + n, i + n, -1.0f0 / rho0
  end
  Ip[n+m+1] = n + m + 1

  # allocate the temporary matrix for combined hessian matrix ###################
  Tp = alloc_mem!(iwork, n + m + 1)
  Ti = alloc_mem!(iwork, Hnnz)
  Tx = alloc_mem!(fwork, Hnnz)

  sptranspose!(ATp, ATi, ATx, Ap, Ai, Ax)
  sphcat!(Tp, Ti, Tx, Pp, Pi, Px, ATp, ATi, ATx)
  Ti = view(Ti, 1:Tp[n+m+1]-1)
  Tx = view(Tx, 1:Tp[n+m+1]-1)
  spmatadd!(Hp, Hi, Hx, Tp, Ti, Tx, Ip, Ii, Ix)
  Hi, Hx = view(Hi, 1:Hp[end]-1), view(Hx, 1:Hp[end]-1)
  sptriu!(Hp, Hi, Hx)

  # allocate and compute the LDLT factorization ################################
  LDLT_etree!(n + m, Hp, Hi, ldlt_iwork, Lnz, info, etree)

  #@cuassert info[1] > 0
  Lnnz = info[1]

  Lp = alloc_mem!(iwork, n + m + 1)
  Li = alloc_mem!(iwork, Lnnz)
  D = alloc_mem!(fwork, n + m)
  Dinv = alloc_mem!(fwork, n + m)
  Lx = alloc_mem!(fwork, Lnnz)

  LDLT_factor!(n + m, (Hp, Hi, Hx), (Lp, Li, Lx), D, Dinv, info, Lnz, etree, ldlt_iwork, ldlt_fwork, false)
  @assert info[1] > 0


  # allocate the ADMM variables ###############################################
  temp = alloc_mem!(fwork, n + m)
  x = alloc_mem!(fwork, n)
  z = alloc_mem!(fwork, m)
  zproj = alloc_mem!(fwork, m)
  y = alloc_mem!(fwork, m)
  v = alloc_mem!(fwork, m)

  # solve in a loop for a fixed number of iterations ##########################
  for i in 1:200
    # x = [sig * x - q z - y ./ rho0]
    @simd for i in 1:n
      @bounds temp[i] = sig * x[i] - q[i]
    end
    @simd for i in 1:m
      @bounds temp[n+i] = z[i] - y[i] / rho0
    end

    # LDLT_solve!(n + m, Lp, Li, Lx, Dinv, x)
    LDLT_solve!(n + m, Lp, Li, Lx, Dinv, temp)

    # x, v = x[1:n], x[n+1:end]
    veccpy!(x, view(temp, 1:n))
    veccpy!(v, view(temp, n+1:n+m))

    # z = z + (v - y) ./ rho0
    @simd for i in 1:m
      @bounds z[i] += (v[i] - y[i]) / rho0
    end

    # zproj = clamp.(z + y ./ rho0, l, u)
    @simd for i in 1:m
      @bounds zproj[i] = z[i] + y[i] / rho0
    end
    vecclamp!(zproj, zproj, l, u)

    # y = y + rho .* (z - zclamped)
    @simd for i in 1:m
      @bounds y[i] += rho0 * (z[i] - zproj[i])
    end

    # debug purposes: compute residuals ###########################
    #rp = 0f0
    #for i in 1:n
    #  rp += (z[i] - zproj[i])^2
    #end
    #rp = sqrt(rp)

    ##spmul!(temp_x, Pp, Pi, Px, x)
    ##spmul!(temp_x2, ATp, ATi, ATx, y)
    #rd = 0f0
    #for i in 1:n
    #  rd += (temp_x[i] + q[i] + temp_x2[i])^2
    #end
    #rd = sqrt(rd)
    #  # (debug) && (@printf("%9.4e - %9.4e\n", rp, rd))

    #  # z = zproj
    #  veccpy!(z, zproj)
  end

  # copy the result into the solution vector ###################################
  veccpy!(view(sol, 1:n), x)
  veccpy!(view(sol, n+1:n+m), y)

  #@cuprintf("length(iwork) = %d\n", len32(iwork)))
  #@cuprintf("length(fwork) = %d\n", len32(fwork)))
  #@cuprintf("used(iwork) = %d\n", Int32(2^12 - len32(iwork)))
  #@cuprintf("used(fwork) = %d\n", Int32(2^12 - len32(fwork)))

  return
end
