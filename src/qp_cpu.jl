import .Threads: @threads

function define_kernel_cpu(config)
  return quote
    function QP_solve_cpu!(
      sols,
      infos,
      iterss,
      ns,
      ms,
      Ps,
      qs,
      As,
      ls,
      us,
      iworks,
      fworks,
      work_sizes,
      offsets,
    )
      rho0, sig = 1.0f1, 1.0f-6

      for idx in 1:length(ns)
        n_offsets, m_offsets, Pnnz_offsets, Annz_offsets, work_offsets = offsets
        (idx > length(ns)) && (return)

        iters, n, m = iterss[idx], ns[idx], ms[idx]
        sol = view(sols, n_offsets[idx]+m_offsets[idx]+1:n_offsets[idx]+m_offsets[idx]+n+m)
        q_ = view(qs, n_offsets[idx]+1:n_offsets[idx]+n)
        l_ = view(ls, m_offsets[idx]+1:m_offsets[idx]+m)
        u_ = view(us, m_offsets[idx]+1:m_offsets[idx]+m)

        Pps, Pis, Pxs = Ps
        Pp_ = view(Pps, n_offsets[idx]+(idx-1)+1:n_offsets[idx]+(idx-1)+n+1)
        Pnnz = Pp_[end] - Pp_[1]
        Pi_ = view(Pis, Pnnz_offsets[idx]+1:Pnnz_offsets[idx]+Pnnz)
        Px_ = view(Pxs, Pnnz_offsets[idx]+1:Pnnz_offsets[idx]+Pnnz)
        P_ = (Pp_, Pi_, Px_)

        Aps, Ais, Axs = As
        Ap_ = view(Aps, n_offsets[idx]+(idx-1)+1:n_offsets[idx]+(idx-1)+n+1)
        Annz = Ap_[end] - Ap_[1]
        Ai_ = view(Ais, Annz_offsets[idx]+1:Annz_offsets[idx]+Annz)
        Ax_ = view(Axs, Annz_offsets[idx]+1:Annz_offsets[idx]+Annz)
        A_ = (Ap_, Ai_, Ax_)

        work_size = work_sizes[idx]
        iwork_ = view(iworks, work_offsets[idx]+1:work_offsets[idx]+work_size)
        fwork_ = view(fworks, work_offsets[idx]+1:work_offsets[idx]+work_size)
        iwork = cpu_make_mem_sf(iwork_, zeros(Int32, 2^12))
        fwork = cpu_make_mem_sf(fwork_, zeros(Float32, 4 * 2^12))
        info_ = view(infos, 5*(idx-1)+1:5*idx)

        imem_fast, fmem_fast, imem_slow, fmem_slow = 0, 0, 0, 0

        # allocate working memory for temporary matrices #############################
        q = cpu_alloc_mem_sf!(fwork, n, $(config[:mem_q]))
        l = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_lu]))
        u = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_lu]))
        veccpy!(q, q_), veccpy!(l, l_), veccpy!(u, u_)

        Pnnz, Annz = len32(P_[2]), len32(A_[2])
        Hnnz = Int32(Pnnz + 2 * Annz + m + n)

        H = (
          cpu_alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_H])),
          cpu_alloc_mem_sf!(iwork, Hnnz, $(config[:mem_H])),
          cpu_alloc_mem_sf!(fwork, Hnnz, $(config[:mem_H])),
        ) # alloc H
        # alloc AT
        AT = (
          cpu_alloc_mem_sf!(iwork, m + 1, $(config[:mem_AT])),
          cpu_alloc_mem_sf!(iwork, Annz, $(config[:mem_AT])),
          cpu_alloc_mem_sf!(fwork, Annz, $(config[:mem_AT])),
        )
        P = (
          cpu_alloc_mem_sf!(iwork, n + 1, $(config[:mem_P])),
          cpu_alloc_mem_sf!(iwork, Pnnz + n, $(config[:mem_P])),
          cpu_alloc_mem_sf!(fwork, Pnnz + n, $(config[:mem_P])),
        ) # alloc P
        I = (
          cpu_alloc_mem_sf!(iwork, n + 1, $(config[:mem_I])),
          cpu_alloc_mem_sf!(iwork, n, $(config[:mem_I])),
          cpu_alloc_mem_sf!(fwork, n, $(config[:mem_I])),
        ) # alloc I

        for i in 1:n
          @cinbounds I[1][i], I[2][i], I[3][i] = i, i, sig
        end
        I[1][n+1] = n + 1
        spmatadd!(P..., P_..., I...)
        P = trim_spmat(P...)
        Pnnz = Int32(P[1][end] - 1)
        cpu_free_mem_sf!(fwork, n, $(config[:mem_I]))
        cpu_free_mem_sf!(iwork, n, $(config[:mem_I]))
        cpu_free_mem_sf!(iwork, n + 1, $(config[:mem_I])) # free I
        #veccpy!(P[1], P_[1]), veccpy!(P[2], P_[2]), veccpy!(P[3], P_[3])

        A = (
          cpu_alloc_mem_sf!(iwork, n + 1, $(config[:mem_A])),
          cpu_alloc_mem_sf!(iwork, Annz, $(config[:mem_A])),
          cpu_alloc_mem_sf!(fwork, Annz, $(config[:mem_A])),
        ) # alloc A
        veccpy!(A[1], A_[1]), veccpy!(A[2], A_[2]), veccpy!(A[3], A_[3])
        sptranspose!(AT..., A...)
        ##################################################################################################

        # create an identity matrix ##################################################
        I = (
          cpu_alloc_mem_sf!(iwork, m + 1, $(config[:mem_I])),
          cpu_alloc_mem_sf!(iwork, m, $(config[:mem_I])),
          cpu_alloc_mem_sf!(fwork, m, $(config[:mem_I])),
        ) # alloc I
        for i in 1:m
          @cinbounds I[1][i], I[2][i], I[3][i] = i, i, -1.0f0 / rho0
        end
        @cinbounds I[1][m+1] = m + 1

        # allocate the temporary matrix for combined hessian matrix ###################
        T_upper = (
          cpu_alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_T])),
          cpu_alloc_mem_sf!(iwork, Pnnz + Annz, $(config[:mem_T])),
          cpu_alloc_mem_sf!(fwork, Pnnz + Annz, $(config[:mem_T])),
        ) # alloc T_upper
        T_lower = (
          cpu_alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_T])),
          cpu_alloc_mem_sf!(iwork, Annz + m, $(config[:mem_T])),
          cpu_alloc_mem_sf!(fwork, Annz + m, $(config[:mem_T])),
        ) # alloc T_lower

        sphcat!(T_upper..., P..., AT...)
        T_upper = trim_spmat(T_upper...)
        sphcat!(T_lower..., A..., I...)
        T_lower = trim_spmat(T_lower...)
        spvcat!(H..., n, T_upper..., T_lower...)
        H = trim_spmat(H...)

        imem_fast = max(imem_fast, fast_buffer_used(iwork))
        fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
        imem_slow = max(imem_slow, slow_buffer_used(iwork))
        fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
        cpu_free_mem_sf!(iwork, n + m + 1, $(config[:mem_T]))
        cpu_free_mem_sf!(iwork, Annz + m, $(config[:mem_T]))
        cpu_free_mem_sf!(fwork, Annz + m, $(config[:mem_T])) # free T_lower
        cpu_free_mem_sf!(iwork, n + m + 1, $(config[:mem_T]))
        cpu_free_mem_sf!(iwork, Pnnz + Annz, $(config[:mem_T]))
        cpu_free_mem_sf!(fwork, Pnnz + Annz, $(config[:mem_T])) # free T_upper

        cpu_free_mem_sf!(iwork, m + 1, $(config[:mem_I]))
        cpu_free_mem_sf!(iwork, m, $(config[:mem_I]))
        cpu_free_mem_sf!(fwork, m, $(config[:mem_I])) # free I
        cpu_free_mem_sf!(fwork, Annz, $(config[:mem_A]))
        cpu_free_mem_sf!(iwork, Annz, $(config[:mem_A]))
        cpu_free_mem_sf!(iwork, m + 1, $(config[:mem_A])) # free A
        cpu_free_mem_sf!(fwork, Pnnz + n, $(config[:mem_P]))
        cpu_free_mem_sf!(iwork, Pnnz + n, $(config[:mem_P]))
        cpu_free_mem_sf!(iwork, n + 1, $(config[:mem_P])) # free P
        cpu_free_mem_sf!(fwork, Annz, $(config[:mem_A]))
        cpu_free_mem_sf!(iwork, Annz, $(config[:mem_A]))
        cpu_free_mem_sf!(iwork, n + 1, $(config[:mem_A])) # free AT

        # compute the permutation and ordering ########################################


        H = trim_spmat(H...)
        Hnnz = H[1][end] - 1
        if $(config[:use_amd] == 1)
          perm = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_perm]))
          n_bits = AMDPKG.compute_n_bits(n + m)
          ordering_iwork = cpu_alloc_mem_sf!(iwork, 2 * (n + m + n_bits), 0)
          AMDPKG.find_ordering(perm, H[1], H[2], ordering_iwork)
          imem_fast = max(imem_fast, fast_buffer_used(iwork))
          fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
          imem_slow = max(imem_slow, slow_buffer_used(iwork))
          fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
          cpu_free_mem_sf!(iwork, 2 * (n + m + n_bits), 0)
          iperm = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_perm]))
          iperm_iwork = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_perm_work]))
          AMDPKG.mergeargsort!(iperm, perm, 1, length(perm), iperm_iwork)
          imem_fast = max(imem_fast, fast_buffer_used(iwork))
          fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
          imem_slow = max(imem_slow, slow_buffer_used(iwork))
          fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
          cpu_free_mem_sf!(iwork, n + m, $(config[:mem_perm_work]))
          H_perm = (
            cpu_alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_H_perm])),
            cpu_alloc_mem_sf!(iwork, Hnnz, $(config[:mem_H_perm])),
            cpu_alloc_mem_sf!(fwork, Hnnz, $(config[:mem_H_perm])),
          )
          permute_iwork = cpu_alloc_mem_sf!(iwork, 2 * (n + m), $(config[:mem_H_perm_work]))
          AMDPKG.permute_mat(H_perm, H, perm, iperm, permute_iwork)
          imem_fast = max(imem_fast, fast_buffer_used(iwork))
          fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
          imem_slow = max(imem_slow, slow_buffer_used(iwork))
          fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
          cpu_free_mem_sf!(iwork, 2 * (n + m), $(config[:mem_H_perm_work]))
          H = H_perm
        end
        sptriu!(H...)

        # allocate and compute the LDLT factorization ################################

        Lnz = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_Lnz]))
        info = cpu_alloc_mem_sf!(iwork, 1, 1)
        etree = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_etree]))
        etree_iwork = cpu_alloc_mem_sf!(iwork, n + m, $(config[:mem_etree_iwork])) # alloc etree_iwork

        LDLT_etree!(n + m, H[1:2]..., etree_iwork, Lnz, info, etree)
        imem_fast = max(imem_fast, fast_buffer_used(iwork))
        fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
        imem_slow = max(imem_slow, slow_buffer_used(iwork))
        fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
        cpu_free_mem_sf!(iwork, n + m, $(config[:mem_etree_iwork])) # free etree_iwork

        if info[1] <= 0 #@assert info[1] > 0
          vecfill!(sol, NaN)
          @printf("Etree failed on idx = %d\n", Int32(idx))
          return
        end
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
        imem_fast = max(imem_fast, fast_buffer_used(iwork))
        fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
        imem_slow = max(imem_slow, slow_buffer_used(iwork))
        fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
        cpu_free_mem_sf!(iwork, 4 * (n + m), $(config[:mem_ldlt_iwork])) # free ldlt_iwork
        cpu_free_mem_sf!(fwork, 2 * (n + m), $(config[:mem_ldlt_fwork])) # free ldlt_fwork
        if info[1] <= 0 # @assert info[1] > 0
          vecfill!(sol, NaN)
          @printf("factor failed on idx = %d\n", Int32(idx))
          return
        end

        # allocate the ADMM variables ###############################################
        temp = cpu_alloc_mem_sf!(fwork, n + m, $(config[:mem_temp]))
        temp2 = cpu_alloc_mem_sf!(fwork, n + m, $(config[:mem_temp]))
        x = cpu_alloc_mem_sf!(fwork, n, $(config[:mem_x]))
        v = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_v]))
        z = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_z]))
        y = cpu_alloc_mem_sf!(fwork, m, $(config[:mem_y]))

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

        # show the memory usage ######################################################
        imem_fast = max(imem_fast, fast_buffer_used(iwork))
        fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
        imem_slow = max(imem_slow, slow_buffer_used(iwork))
        fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
        info_[1] = Int32(imem_fast)
        info_[2] = Int32(fmem_fast)
        info_[3] = Int32(imem_slow)
        info_[4] = Int32(fmem_slow)
        info_[5] = Int32(k)
      end
      return
    end
  end
end
