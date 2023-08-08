function define_kernel(memory_mask)
  @assert length(memory_mask) >= 22

  mem_AT, mem_A, mem_P, mem_q, mem_lu, mem_H, mem_I, mem_T = memory_mask[1:8]
  mem_Lnz, mem_etree, mem_etree_iwork, mem_ldlt_iwork, mem_ldlt_fwork = memory_mask[9:13]
  mem_L, mem_D, mem_temp, mem_x, mem_z, mem_zproj, mem_y, mem_v = memory_mask[14:22]

  return quote
    function QP_solve!(sol, info_, iters, n, m, P_, q_, A_, l_, u_, iwork_, fwork_)
      #(threadIdx().x != 1 || blockIdx().x != 1) && (return)

      imem, fmem = 0, 0

      iwork = make_mem_sf(iwork_, CuDynamicSharedArray(Int32, 2^12))
      fwork = make_mem_sf(fwork_, CuDynamicSharedArray(Float32, 2^12, 4 * 2^12))

      # allocate working memory for temporary matrices #############################
      Pnnz, Annz = len32(P_[2]), len32(A_[2])

      # alloc AT
      AT = (
        alloc_mem_sf!(iwork, m + 1, $mem_AT),
        alloc_mem_sf!(iwork, Annz, $mem_AT),
        alloc_mem_sf!(fwork, Annz, $mem_AT),
      )
      P = (
        alloc_mem_sf!(iwork, n + 1, $mem_P),
        alloc_mem_sf!(iwork, Pnnz, $mem_P),
        alloc_mem_sf!(fwork, Pnnz, $mem_P),
      ) # alloc P
      A = (
        alloc_mem_sf!(iwork, n + 1, $mem_A),
        alloc_mem_sf!(iwork, Annz, $mem_A),
        alloc_mem_sf!(fwork, Annz, $mem_A),
      ) # alloc A
      veccpy!(P[1], P_[1]), veccpy!(P[2], P_[2]), veccpy!(P[3], P_[3])
      veccpy!(A[1], A_[1]), veccpy!(A[2], A_[2]), veccpy!(A[3], A_[3])
      #P, A = P_, A_

      sptranspose!(AT..., A...)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      free_mem_sf!(fwork, Annz, $mem_A)
      free_mem_sf!(iwork, Annz, $mem_A)
      free_mem_sf!(iwork, m + 1, $mem_A) # free A

      sptriu!(P...)
      P = trim_spmat(P...)
      Pnnz = len32(P[2])
      Hnnz = Int32(Pnnz + Annz + m + n)

      q = alloc_mem_sf!(fwork, n, $mem_q)
      l, u = alloc_mem_sf!(fwork, m, $mem_lu), alloc_mem_sf!(fwork, m, $mem_lu)
      veccpy!(q, q_), veccpy!(l, l_), veccpy!(u, u_)
      ##################################################################################################

      rho0, sig = 1.0f1, 1.0f-6

      H = (
        alloc_mem_sf!(iwork, n + m + 1, $mem_H),
        alloc_mem_sf!(iwork, Hnnz, $mem_H),
        alloc_mem_sf!(fwork, Hnnz, $mem_H),
      ) # alloc H

      # create an identity matrix ##################################################
      I = (
        alloc_mem_sf!(iwork, n + m + 1, $mem_I),
        alloc_mem_sf!(iwork, n + m, $mem_I),
        alloc_mem_sf!(fwork, n + m, $mem_I),
      ) # alloc I

      @simd for i in 1:n
        @cinbounds I[1][i], I[2][i], I[3][i] = i, i, sig
      end
      @simd for i in 1:m
        @cinbounds I[1][n+i], I[2][n+i], I[3][n+i] = i + n, i + n, -1.0f0 / rho0
      end
      @cinbounds I[1][n+m+1] = n + m + 1

      # allocate the temporary matrix for combined hessian matrix ###################
      T = (
        alloc_mem_sf!(iwork, n + m + 1, $mem_T),
        alloc_mem_sf!(iwork, Hnnz, $mem_T),
        alloc_mem_sf!(fwork, Hnnz, $mem_T),
      ) # alloc T

      sphcat!(T..., P..., AT...)
      T = trim_spmat(T...)
      spmatadd!(H..., T..., I...)
      #imem[2], fmem[2] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      free_mem_sf!(iwork, n + m + 1, $mem_T)
      free_mem_sf!(iwork, Hnnz, $mem_T)
      free_mem_sf!(fwork, Hnnz, $mem_T) # free T
      free_mem_sf!(iwork, n + m + 1, $mem_I),
      free_mem_sf!(iwork, n + m, $mem_I),
      free_mem_sf!(fwork, n + m, $mem_I) # free I
      #imem[3], fmem[3] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory
      H = trim_spmat(H...)

      # allocate and compute the LDLT factorization ################################

      Lnz = alloc_mem_sf!(iwork, n + m, $mem_Lnz)
      info = alloc_mem_sf!(iwork, 1, 1)
      etree = alloc_mem_sf!(iwork, n + m, $mem_etree)
      etree_iwork = alloc_mem_sf!(iwork, n + m, $mem_etree_iwork) # alloc etree_iwork

      LDLT_etree!(n + m, H[1:2]..., etree_iwork, Lnz, info, etree)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      free_mem_sf!(iwork, n + m, $mem_etree_iwork) # free etree_iwork

      @cuassert info[1] > 0
      Lnnz = info[1]

      L = (
        alloc_mem_sf!(iwork, n + m + 1, $mem_L),
        alloc_mem_sf!(iwork, Lnnz, $mem_L),
        alloc_mem_sf!(fwork, Lnnz, $mem_L),
      ) # alloc L
      D, Dinv = alloc_mem_sf!(fwork, n + m, $mem_D), alloc_mem_sf!(fwork, n + m, $mem_D) # alloc D, Dinv

      ldlt_iwork = alloc_mem_sf!(iwork, 4 * (n + m), $mem_ldlt_iwork) # alloc ldlt_iwork
      ldlt_fwork = alloc_mem_sf!(fwork, 2 * (n + m), $mem_ldlt_iwork) # alloc ldlt_fwork
      #imem[4], fmem[4] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory

      #LDLT_factor!(n + m, (Hp, Hi, Hx), (Lp, Li, Lx), D, Dinv, info, Lnz, etree, ldlt_iwork, ldlt_fwork, false)
      LDLT_factor!(n + m, H, L, D, Dinv, info, Lnz, etree, ldlt_iwork, ldlt_fwork, false)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      free_mem_sf!(iwork, 4 * (n + m), $mem_ldlt_iwork) # free ldlt_iwork
      free_mem_sf!(fwork, 2 * (n + m), $mem_ldlt_fwork) # free ldlt_fwork
      @cuassert info[1] > 0

      # allocate the ADMM variables ###############################################
      temp = alloc_mem_sf!(fwork, n + m, $mem_temp)
      x = alloc_mem_sf!(fwork, n, $mem_x)
      z = alloc_mem_sf!(fwork, m, $mem_z)
      #zproj = alloc_mem_sf!(fwork, m, $mem_zproj)
      y, v = alloc_mem_sf!(fwork, m, $mem_y), alloc_mem_sf!(fwork, m, $mem_v)
      #imem[5], fmem[5] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory

      # solve in a loop for a fixed number of iterations ##########################
      k = 0
      for i in Int32(1):Int32(iters)
        k += 1
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
        #@simd for i in 1:m
        #  @cinbounds zproj[i] = z[i] + y[i] / rho0
        #end
        #vecclamp!(zproj, zproj, l, u)

        ## y = y + rho .* (z - zclamped)
        #@simd for i in 1:m
        #  @cinbounds y[i] += rho0 * (z[i] - zproj[i])
        #end
        # y = y + rho .* (z - zclamped)
        @simd for i in 1:m
          @cinbounds y[i] += rho0 * (z[i] - max(min(z[i] + y[i] / rho0, u[i]), l[i]))
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
      veccpy!(view(sol, 1:n), x), veccpy!(view(sol, n+1:n+m), y)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      info_[1] = Int32(imem)
      info_[2] = Int32(fmem)
      info_[3] = Int32(k)

      return
    end
  end
end
