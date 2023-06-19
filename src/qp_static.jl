function QP_solve!(sol, n, m, P_, q_, A_, l_, u_)
  #(threadIdx().x != 1 || blockIdx().x != 1) && (return)

  imem, fmem = view(sol, 1:5), view(sol, 6:10)

  iwork = make_mem(CuDynamicSharedArray(Int32, 2^12))
  fwork = make_mem(CuDynamicSharedArray(Float32, 2^12, 4 * length(iwork)))

  # allocate working memory for temporary matrices #############################
  #Pnnz, Annz = len32(P_[2]), len32(A_[2])
  Pnnz, Annz = len32(P_[2]), len32(A_[2])

  AT = (alloc_mem!(iwork, m + 1), alloc_mem!(iwork, Annz), alloc_mem!(fwork, Annz)) # alloc AT

  #P = (alloc_mem!(iwork, n + 1), alloc_mem!(iwork, Pnnz), alloc_mem!(fwork, Pnnz)) # alloc P
  #A = (alloc_mem!(iwork, n + 1), alloc_mem!(iwork, Annz), alloc_mem!(fwork, Annz)) # alloc A
  #veccpy!(P[1], P_[1]), veccpy!(P[2], P_[2]), veccpy!(P[3], P_[3])
  #veccpy!(A[1], A_[1]), veccpy!(A[2], A_[2]), veccpy!(A[3], A_[3])
  P, A = P_, A_

  imem[1], fmem[1] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory
  sptranspose!(AT..., A...)
  #free_mem!(fwork, Annz), free_mem!(iwork, Annz), free_mem!(iwork, m + 1) # free A

  sptriu!(P...)
  P = trim_spmat(P...)
  Pnnz = len32(P[2])
  Hnnz = Int32(Pnnz + Annz + m + n)

  q, l, u = alloc_mem!(fwork, n), alloc_mem!(fwork, m), alloc_mem!(fwork, m)
  veccpy!(q, q_), veccpy!(l, l_), veccpy!(u, u_)
  ##################################################################################################

  rho0, sig = 1.0f1, 1.0f-4

  H = (alloc_mem!(iwork, n + m + 1), alloc_mem!(iwork, Hnnz), alloc_mem!(fwork, Hnnz)) # alloc H

  # create an identity matrix ##################################################
  I = (alloc_mem!(iwork, n + m + 1), alloc_mem!(iwork, n + m), alloc_mem!(fwork, n + m)) # alloc I

  @simd for i in 1:n
    @cinbounds I[1][i], I[2][i], I[3][i] = i, i, sig
  end
  @simd for i in 1:m
    @cinbounds I[1][n+i], I[2][n+i], I[3][n+i] = i + n, i + n, -1.0f0 / rho0
  end
  @cinbounds I[1][n+m+1] = n + m + 1

  # allocate the temporary matrix for combined hessian matrix ###################
  T = (alloc_mem!(iwork, n + m + 1), alloc_mem!(iwork, Hnnz), alloc_mem!(fwork, Hnnz)) # alloc T

  sphcat!(T..., P..., AT...)
  T = trim_spmat(T...)
  spmatadd!(H..., T..., I...)
  imem[2], fmem[2] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory
  free_mem!(iwork, n + m + 1), free_mem!(iwork, Hnnz), free_mem!(fwork, Hnnz) # free T
  free_mem!(iwork, n + m + 1), free_mem!(iwork, n + m), free_mem!(fwork, n + m) # free I
  imem[3], fmem[3] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory
  H = trim_spmat(H...)

  # allocate and compute the LDLT factorization ################################

  Lnz = alloc_mem!(iwork, n + m)
  info = alloc_mem!(iwork, 1)
  etree = alloc_mem!(iwork, n + m)
  etree_iwork = alloc_mem!(iwork, n + m) # alloc etree_iwork
  
  LDLT_etree!(n + m, H[1:2]..., etree_iwork, Lnz, info, etree)
  free_mem!(iwork, n + m) # free etree_iwork

  @cuassert info[1] > 0
  Lnnz = info[1]

  L = alloc_mem!(iwork, n + m + 1), alloc_mem!(iwork, Lnnz), alloc_mem!(fwork, Lnnz) # alloc L
  D, Dinv = alloc_mem!(fwork, n + m), alloc_mem!(fwork, n + m) # alloc D, Dinv

  ldlt_iwork = alloc_mem!(iwork, 4 * (n + m)) # alloc ldlt_iwork
  ldlt_fwork = alloc_mem!(fwork, 2 * (n + m)) # alloc ldlt_fwork
  imem[4], fmem[4] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory

  #LDLT_factor!(n + m, (Hp, Hi, Hx), (Lp, Li, Lx), D, Dinv, info, Lnz, etree, ldlt_iwork, ldlt_fwork, false)
  LDLT_factor!(n + m, H, L, D, Dinv, info, Lnz, etree, ldlt_iwork, ldlt_fwork, false)
  free_mem!(iwork, 4 * (n + m)) # free ldlt_iwork
  free_mem!(fwork, 2 * (n + m)) # free ldlt_fwork
  @cuassert info[1] > 0

  # allocate the ADMM variables ###############################################
  temp = alloc_mem!(fwork, n + m)
  x = alloc_mem!(fwork, n)
  z = alloc_mem!(fwork, m)
  zproj = alloc_mem!(fwork, m)
  y = alloc_mem!(fwork, m)
  v = alloc_mem!(fwork, m)
  imem[5], fmem[5] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory

  # solve in a loop for a fixed number of iterations ##########################
  for i in 1:200
    # x = [sig * x - q z - y ./ rho0]
    @simd for i in 1:n
      @cinbounds temp[i] = sig * x[i] - q[i]
    end
    @simd for i in 1:m
      @cinbounds temp[n+i] = z[i] - y[i] / rho0
    end

    # LDLT_solve!(n + m, Lp, Li, Lx, Dinv, x)
    LDLT_solve!(n + m, L..., Dinv, temp)

    # x, v = x[1:n], x[n+1:end]
    veccpy!(x, view(temp, 1:n)), veccpy!(v, view(temp, n+1:n+m))

    # z = z + (v - y) ./ rho0
    @simd for i in 1:m
      @cinbounds z[i] += (v[i] - y[i]) / rho0
    end

    # zproj = clamp.(z + y ./ rho0, l, u)
    @simd for i in 1:m
      @cinbounds zproj[i] = z[i] + y[i] / rho0
    end
    vecclamp!(zproj, zproj, l, u)

    # y = y + rho .* (z - zclamped)
    @simd for i in 1:m
      @cinbounds y[i] += rho0 * (z[i] - zproj[i])
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

  if true
    max_imem, max_fmem = 0, 0
    for i in 1:5
      @cuprintf("(%04d, %04d)\n", Int32(imem[i]), Int32(fmem[i]))
      max_imem = max(max_imem, imem[i])
      max_fmem = max(max_fmem, fmem[i])
    end
    @cuprintf("\nmax = (%04d, %04d)\n\n", Int32(max_imem), Int32(max_fmem))
  end

  veccpy!(view(sol, 1:n), x), veccpy!(view(sol, n+1:n+m), y)

  #@cuprintf("length(iwork) = %d\n", len32(iwork)))
  #@cuprintf("length(fwork) = %d\n", len32(fwork)))
  #@cuprintf("used(iwork) = %d\n", Int32(2^12 - len32(iwork)))
  #@cuprintf("used(fwork) = %d\n", Int32(2^12 - len32(fwork)))

  return
end
