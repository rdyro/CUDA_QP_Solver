function define_kernel_cpu(config)
  return quote
    function QP_solve_cpu!(sol, info_, iters, n, m, P_, q_, A_, l_, u_, iwork_, fwork_)
      imem, fmem = 0, 0

      iwork = cpu_make_mem_sf(iwork_, zeros(Int32, 2^12))
      fwork = cpu_make_mem_sf(fwork_, zeros(Float32, 4 * 2^12))

      # allocate working memory for temporary matrices #############################
      Pnnz, Annz = len32(P_[2]), len32(A_[2])

      # alloc AT
      AT = (
        cpu_alloc_mem_sf!(iwork, m + 1, $(config[:mem_AT])),
        cpu_alloc_mem_sf!(iwork, Annz, $(config[:mem_AT])),
        cpu_alloc_mem_sf!(fwork, Annz, $(config[:mem_AT])),
      )
      P = (
        cpu_alloc_mem_sf!(iwork, n + 1, $(config[:mem_P])),
        cpu_alloc_mem_sf!(iwork, Pnnz, $(config[:mem_P])),
        cpu_alloc_mem_sf!(fwork, Pnnz, $(config[:mem_P])),
      ) # alloc P
      veccpy!(P[1], P_[1]), veccpy!(P[2], P_[2]), veccpy!(P[3], P_[3])
      A = (
        cpu_alloc_mem_sf!(iwork, n + 1, $(config[:mem_A])),
        cpu_alloc_mem_sf!(iwork, Annz, $(config[:mem_A])),
        cpu_alloc_mem_sf!(fwork, Annz, $(config[:mem_A])),
      ) # alloc A
      veccpy!(A[1], A_[1]), veccpy!(A[2], A_[2]), veccpy!(A[3], A_[3])

      sptranspose!(AT..., A...)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      cpu_free_mem_sf!(fwork, Annz, $(config[:mem_A]))
      cpu_free_mem_sf!(iwork, Annz, $(config[:mem_A]))
      cpu_free_mem_sf!(iwork, m + 1, $(config[:mem_A])) # free A

      sptriu!(P...)
      P = trim_spmat(P...)
      Pnnz = len32(P[2])
      Hnnz = Int32(Pnnz + Annz + m + n)

      q = cpu_alloc_mem_sf!(fwork, n, $(config[:mem_q]))
      l = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_lu]))
      u = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_lu]))
      veccpy!(q, q_), veccpy!(l, l_), veccpy!(u, u_)
      ##################################################################################################

      rho0, sig = 1.0f1, 1.0f-6

      H = (
        cpu_alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_H])),
        cpu_alloc_mem_sf!(iwork, Hnnz, $(config[:mem_H])),
        cpu_alloc_mem_sf!(fwork, Hnnz, $(config[:mem_H])),
      ) # alloc H

      # create an identity matrix ##################################################
      I = (
        cpu_alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_I])),
        cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_I])),
        cpu_alloc_mem_sf!(fwork, n + m, $(config[:mem_I])),
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
        cpu_alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_T])),
        cpu_alloc_mem_sf!(iwork, Hnnz, $(config[:mem_T])),
        cpu_alloc_mem_sf!(fwork, Hnnz, $(config[:mem_T])),
      ) # alloc T

      sphcat!(T..., P..., AT...)
      T = trim_spmat(T...)
      spmatadd!(H..., T..., I...)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      cpu_free_mem_sf!(iwork, n + m + 1, $(config[:mem_T]))
      cpu_free_mem_sf!(iwork, Hnnz, $(config[:mem_T]))
      cpu_free_mem_sf!(fwork, Hnnz, $(config[:mem_T])) # free T
      cpu_free_mem_sf!(iwork, n + m + 1, $(config[:mem_I]))
      cpu_free_mem_sf!(iwork, n + m, $(config[:mem_I]))
      cpu_free_mem_sf!(fwork, n + m, $(config[:mem_I])) # free I

      # compute the permutation and ordering ########################################

      if $(config[:use_amd] == 1)
        perm = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_perm]))
        n_bits = AMDPKG.compute_n_bits(n + m)
        ordering_iwork = cpu_alloc_mem_sf!(iwork, 2 * (n + m + n_bits), 0)
        AMDPKG.find_ordering(perm, H[1], H[2], ordering_iwork)
        cpu_free_mem_sf!(iwork, 2 * (n + m + n_bits), 0)
        iperm = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_perm]))
        iperm_iwork = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_perm_work]))
        AMDPKG.mergeargsort!(iperm, perm, 1, length(perm), iperm_iwork)
        imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
        cpu_free_mem_sf!(iwork, n + m, $(config[:mem_perm_work]))
        H_perm = (
          cpu_alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_H_perm])),
          cpu_alloc_mem_sf!(iwork, Hnnz, $(config[:mem_H_perm])),
          cpu_alloc_mem_sf!(fwork, Hnnz, $(config[:mem_H_perm])),
        )
        permute_iwork = cpu_alloc_mem_sf!(iwork, 2 * (n + m), $(config[:mem_H_perm_work]))
        imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
        AMDPKG.permute_mat(H_perm, H, perm, iperm, permute_iwork)
        cpu_free_mem_sf!(iwork, 2 * (n + m), $(config[:mem_H_perm_work]))
        H = H_perm
      end

      H = trim_spmat(H...)

      # allocate and compute the LDLT factorization ################################

      Lnz = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_Lnz]))
      info = cpu_alloc_mem_sf!(iwork, 1, 1)
      etree = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_etree]))
      etree_iwork = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_etree_iwork])) # alloc etree_iwork

      LDLT_etree!(n + m, H[1:2]..., etree_iwork, Lnz, info, etree)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      cpu_free_mem_sf!(iwork, n + m, $(config[:mem_etree_iwork])) # free etree_iwork

      @assert info[1] > 0
      Lnnz = info[1]

      L = (
        cpu_alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_L])),
        cpu_alloc_mem_sf!(iwork, Lnnz, $(config[:mem_L])),
        cpu_alloc_mem_sf!(fwork, Lnnz, $(config[:mem_L])),
      ) # alloc L
      D = cpu_alloc_mem_sf!(fwork, n + m, $(config[:mem_D]))
      Dinv = cpu_alloc_mem_sf!(fwork, n + m, $(config[:mem_D])) # alloc D, Dinv

      ldlt_iwork = cpu_alloc_mem_sf!(iwork, 4 * (n + m), $(config[:mem_ldlt_iwork])) # alloc ldlt_iwork
      ldlt_fwork = cpu_alloc_mem_sf!(fwork, 2 * (n + m), $(config[:mem_ldlt_iwork])) # alloc ldlt_fwork

      LDLT_factor!(n + m, H, L, D, Dinv, info, Lnz, etree, ldlt_iwork, ldlt_fwork, false)
      imem, fmem = max(imem, fast_buffer_used(iwork)), max(fmem, fast_buffer_used(fwork))
      cpu_free_mem_sf!(iwork, 4 * (n + m), $(config[:mem_ldlt_iwork])) # free ldlt_iwork
      cpu_free_mem_sf!(fwork, 2 * (n + m), $(config[:mem_ldlt_fwork])) # free ldlt_fwork
      @assert info[1] > 0

      # allocate the ADMM variables ###############################################
      temp = cpu_alloc_mem_sf!(fwork, n + m, $(config[:mem_temp]))
      temp2 = cpu_alloc_mem_sf!(fwork, n + m, $(config[:mem_temp]))
      x = cpu_alloc_mem_sf!(fwork, n, $(config[:mem_x]))
      z = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_z]))
      y = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_y]))
      v = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_v]))

      # solve in a loop for a fixed number of iterations ##########################
      k = 0
      for it in Int32(1):Int32(iters)
        k += 1
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
