function define_kernel_cuda(config)
  return quote
    function QP_solve_cuda!(sol, info_, iters, n, m, P_, q_, A_, l_, u_, iwork_, fwork_)
      #(threadIdx().x != 1 || blockIdx().x != 1) && (return)

      imem, fmem = 0, 0

      iwork = make_mem_sf(iwork_, CuDynamicSharedArray(Int32, 2^12))
      fwork = make_mem_sf(fwork_, CuDynamicSharedArray(Float32, 2^12, 4 * 2^12))
      #iwork = make_mem_sf(iwork_, zeros(Int32, 2^12))
      #fwork = make_mem_sf(fwork_, zeros(Float32, 4 * 2^12))


      # allocate working memory for temporary matrices #############################
      Pnnz, Annz = len32(P_[2]), len32(A_[2])

      # alloc AT
      AT = (
        alloc_mem_sf!(iwork, m + 1, $(config[:mem_AT])),
        alloc_mem_sf!(iwork, Annz, $(config[:mem_AT])),
        alloc_mem_sf!(fwork, Annz, $(config[:mem_AT])),
      )
      P = (
        alloc_mem_sf!(iwork, n + 1, $(config[:mem_P])),
        alloc_mem_sf!(iwork, Pnnz, $(config[:mem_P])),
        alloc_mem_sf!(fwork, Pnnz, $(config[:mem_P])),
      ) # alloc P
      veccpy!(P[1], P_[1]), veccpy!(P[2], P_[2]), veccpy!(P[3], P_[3])
      A = (
        alloc_mem_sf!(iwork, n + 1, $(config[:mem_A])),
        alloc_mem_sf!(iwork, Annz, $(config[:mem_A])),
        alloc_mem_sf!(fwork, Annz, $(config[:mem_A])),
      ) # alloc A
      veccpy!(A[1], A_[1]), veccpy!(A[2], A_[2]), veccpy!(A[3], A_[3])

      sptranspose!(AT..., A...)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      free_mem_sf!(fwork, Annz, $(config[:mem_A]))
      free_mem_sf!(iwork, Annz, $(config[:mem_A]))
      free_mem_sf!(iwork, m + 1, $(config[:mem_A])) # free A

      sptriu!(P...)
      P = trim_spmat(P...)
      Pnnz = len32(P[2])
      Hnnz = Int32(Pnnz + Annz + m + n)

      q = alloc_mem_sf!(fwork, n, $(config[:mem_q]))
      l = alloc_mem_sf!(fwork, m, $(config[:mem_lu]))
      u = alloc_mem_sf!(fwork, m, $(config[:mem_lu]))
      veccpy!(q, q_), veccpy!(l, l_), veccpy!(u, u_)
      ##################################################################################################

      rho0, sig = 1.0f1, 1.0f-6

      H = (
        alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_H])),
        alloc_mem_sf!(iwork, Hnnz, $(config[:mem_H])),
        alloc_mem_sf!(fwork, Hnnz, $(config[:mem_H])),
      ) # alloc H

      # create an identity matrix ##################################################
      I = (
        alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_I])),
        alloc_mem_sf!(iwork, n + m, $(config[:mem_I])),
        alloc_mem_sf!(fwork, n + m, $(config[:mem_I])),
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
        alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_T])),
        alloc_mem_sf!(iwork, Hnnz, $(config[:mem_T])),
        alloc_mem_sf!(fwork, Hnnz, $(config[:mem_T])),
      ) # alloc T

      sphcat!(T..., P..., AT...)
      T = trim_spmat(T...)
      spmatadd!(H..., T..., I...)
      #imem[2], fmem[2] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      free_mem_sf!(iwork, n + m + 1, $(config[:mem_T]))
      free_mem_sf!(iwork, Hnnz, $(config[:mem_T]))
      free_mem_sf!(fwork, Hnnz, $(config[:mem_T])) # free T
      free_mem_sf!(iwork, n + m + 1, $(config[:mem_I]))
      free_mem_sf!(iwork, n + m, $(config[:mem_I]))
      free_mem_sf!(fwork, n + m, $(config[:mem_I])) # free I
      #imem[3], fmem[3] = 2^12 - length(iwork), 2^12 - length(fwork) # save remaining memory

      # compute the permutation and ordering ########################################

      if $(config[:use_amd] == 1)
        perm = alloc_mem_sf!(iwork, n + m, $(config[:mem_perm]))
        n_bits = AMDPKG.compute_n_bits(n + m)
        ordering_iwork = alloc_mem_sf!(iwork, 2 * (n + m + n_bits), 0)
        AMDPKG.find_ordering(perm, H[1], H[2], ordering_iwork)
        free_mem_sf!(iwork, 2 * (n + m + n_bits), 0)
        iperm = alloc_mem_sf!(iwork, n + m, $(config[:mem_perm]))
        iperm_iwork = alloc_mem_sf!(iwork, n + m, $(config[:mem_perm_work]))
        AMDPKG.mergeargsort!(iperm, perm, 1, length(perm), iperm_iwork)
        imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
        free_mem_sf!(iwork, n + m, $(config[:mem_perm_work]))
        H_perm = (
          alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_H_perm])),
          alloc_mem_sf!(iwork, Hnnz, $(config[:mem_H_perm])),
          alloc_mem_sf!(fwork, Hnnz, $(config[:mem_H_perm])),
        )
        permute_iwork = alloc_mem_sf!(iwork, 2 * (n + m), $(config[:mem_H_perm_work]))
        imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
        AMDPKG.permute_mat(H_perm, H, perm, iperm, permute_iwork)
        free_mem_sf!(iwork, 2 * (n + m), $(config[:mem_H_perm_work]))
        H = H_perm
      end

      H = trim_spmat(H...)

      # allocate and compute the LDLT factorization ################################

      Lnz = alloc_mem_sf!(iwork, n + m, $(config[:mem_Lnz]))
      info = alloc_mem_sf!(iwork, 1, 1)
      etree = alloc_mem_sf!(iwork, n + m, $(config[:mem_etree]))
      etree_iwork = alloc_mem_sf!(iwork, n + m, $(config[:mem_etree_iwork])) # alloc etree_iwork

      LDLT_etree!(n + m, H[1:2]..., etree_iwork, Lnz, info, etree)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      free_mem_sf!(iwork, n + m, $(config[:mem_etree_iwork])) # free etree_iwork

      #@cuassert info[1] > 0
      Lnnz = info[1]

      L = (
        alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_L])),
        alloc_mem_sf!(iwork, Lnnz, $(config[:mem_L])),
        alloc_mem_sf!(fwork, Lnnz, $(config[:mem_L])),
      ) # alloc L
      D = alloc_mem_sf!(fwork, n + m, $(config[:mem_D]))
      Dinv = alloc_mem_sf!(fwork, n + m, $(config[:mem_D])) # alloc D, Dinv

      ldlt_iwork = alloc_mem_sf!(iwork, 4 * (n + m), $(config[:mem_ldlt_iwork])) # alloc ldlt_iwork
      ldlt_fwork = alloc_mem_sf!(fwork, 2 * (n + m), $(config[:mem_ldlt_iwork])) # alloc ldlt_fwork

      LDLT_factor!(n + m, H, L, D, Dinv, info, Lnz, etree, ldlt_iwork, ldlt_fwork, false)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      free_mem_sf!(iwork, 4 * (n + m), $(config[:mem_ldlt_iwork])) # free ldlt_iwork
      free_mem_sf!(fwork, 2 * (n + m), $(config[:mem_ldlt_fwork])) # free ldlt_fwork
      #@cuassert info[1] > 0

      # allocate the ADMM variables ###############################################
      temp = alloc_mem_sf!(fwork, n + m, $(config[:mem_temp]))
      temp2 = alloc_mem_sf!(fwork, n + m, $(config[:mem_temp]))
      x = alloc_mem_sf!(fwork, n, $(config[:mem_x]))
      z = alloc_mem_sf!(fwork, m, $(config[:mem_z]))
      y = alloc_mem_sf!(fwork, m, $(config[:mem_y]))
      v = alloc_mem_sf!(fwork, m, $(config[:mem_v]))

      # solve in a loop for a fixed number of iterations ##########################
      k = 0
      for i in Int32(1):Int32(iters)
        k += 1
        #@simd for i in 1:n
        #  @cinbounds temp[i] = sig * x[i] - q[i]
        #end
        #@simd for i in 1:m
        #  @cinbounds temp[n+i] = z[i] - y[i] / rho0
        #end
        #temp[1:n] .= sig * x - q
        #temp[n+1:n+m] .= z - y ./ rho0
        admm_set_rhs_top!(view(temp, 1:n), sig, x, q)
        admm_set_rhs_bot!(view(temp, n+1:n+m), z, y, rho0)


        # solve the problem
        if $(config[:use_amd] == 1)
          vecpermute!(temp2, temp, perm)
          LDLT_solve!(n + m, L..., Dinv, temp2)
          vecpermute!(temp, temp2, iperm)
        else
          LDLT_solve!(n + m, L..., Dinv, temp)
        end

        veccpy!(x, view(temp, 1:n))
        veccpy!(v, view(temp, n+1:n+m))
        admm_update_z!(z, v, y, rho0)
        admm_update_y!(y, z, l, u, rho0)
        #@simd for i in 1:m
        #  @cinbounds z[i] += (v[i] - y[i]) / rho0
        #end
        #@simd for i in 1:m
        #  @cinbounds y[i] += rho0 * (z[i] - max(min(z[i] + y[i] / rho0, u[i]), l[i]))
        #end
      end

      # copy the result into the solution vector ###################################
      veccpy!(view(sol, 1:n), x)
      veccpy!(view(sol, n+1:n+m), y)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      info_[1] = Int32(imem)
      info_[2] = Int32(fmem)
      info_[3] = Int32(k)
      return
    end
  end
end
