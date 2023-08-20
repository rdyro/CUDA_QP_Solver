function cuprint_matrix(H)
  H = trim_spmat(H...)
  Hp, Hi, Hx = H
  @cuprintf("{\"Ap\": [")
  for i in 1:length(Hp)-1
    @cuprintf("%d, ", Int32(Hp[i]))
  end
  @cuprintf("%d], \"Ai\": [", Int32(Hp[end]))
  Hnnz = Hp[end] - 1
  for i in 1:Hnnz-1
    @cuprintf("%d, ", Int32(Hi[i]))
  end
  @cuprintf("%d], \"Ax\": [", Int32(Hi[Hnnz]))
  for i in 1:Hnnz-1
    @cuprintf("%f, ", Hx[i])
  end
  @cuprintf("%f]}\n", Hx[Hnnz])
  return
end

function define_kernel_cuda(config)
  return quote
    function QP_solve_cuda!(
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
      n_threads,
    )
      rho0, sig = 1.0f1, 1.0f-6

      n_offsets, m_offsets, Pnnz_offsets, Annz_offsets, work_offsets = offsets

      thread_idx = threadIdx().x
      idx = blockIdx().x
      if idx > length(ns)
        return
      end

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
      #iwork_ = view(iworks, work_offsets[idx]+1:work_offsets[idx]+work_size)::SubArray{Int32,1,CuDeviceVector{Int32,1},Tuple{UnitRange{Int64}},true}
      iwork_ = view(iworks, work_offsets[idx]+1:work_offsets[idx]+work_size)
      fwork_ = view(fworks, work_offsets[idx]+1:work_offsets[idx]+work_size)
      iwork = make_mem_sf(iwork_, CuDynamicSharedArray(Int32, 2^12))
      fwork =
        make_mem_sf(fwork_, CuDynamicSharedArray(Float32, length(iwork.fast_buffer), 4 * 2^12))
      info_ = view(infos, 5*(idx-1)+1:5*idx)

      imem_fast, fmem_fast, imem_slow, fmem_slow = 0, 0, 0, 0

      # allocate working memory for temporary matrices #############################
      q = alloc_mem_sf!(fwork, n, $(config[:mem_q]))
      l = alloc_mem_sf!(fwork, m, $(config[:mem_lu]))
      u = alloc_mem_sf!(fwork, m, $(config[:mem_lu]))
      if thread_idx == 1
        veccpy!(q, q_), veccpy!(l, l_), veccpy!(u, u_)
      end

      Pnnz, Annz = len32(P_[2]), len32(A_[2])
      Hnnz = Int32(Pnnz + 2 * Annz + m + n)

      H = (
        alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_H])),
        alloc_mem_sf!(iwork, Hnnz, $(config[:mem_H])),
        alloc_mem_sf!(fwork, Hnnz, $(config[:mem_H])),
      ) # alloc H
      # alloc AT
      AT = (
        alloc_mem_sf!(iwork, m + 1, $(config[:mem_AT])),
        alloc_mem_sf!(iwork, Annz, $(config[:mem_AT])),
        alloc_mem_sf!(fwork, Annz, $(config[:mem_AT])),
      )
      P = (
        alloc_mem_sf!(iwork, n + 1, $(config[:mem_P])),
        alloc_mem_sf!(iwork, Pnnz + n, $(config[:mem_P])),
        alloc_mem_sf!(fwork, Pnnz + n, $(config[:mem_P])),
      ) # alloc P
      I = (
        alloc_mem_sf!(iwork, n + 1, $(config[:mem_I])),
        alloc_mem_sf!(iwork, n, $(config[:mem_I])),
        alloc_mem_sf!(fwork, n, $(config[:mem_I])),
      ) # alloc I

      if thread_idx == 1
        for i in 1:n
          @cinbounds I[1][i], I[2][i], I[3][i] = i, i, sig
        end
        I[1][n+1] = n + 1
        spmatadd!(P..., P_..., I...)
        P = trim_spmat(P...)
      end
      sync_threads()
      Pnnz = Int32(P[1][end] - 1)
      free_mem_sf!(fwork, n, $(config[:mem_I]))
      free_mem_sf!(iwork, n, $(config[:mem_I]))
      free_mem_sf!(iwork, n + 1, $(config[:mem_I])) # free I
      #veccpy!(P[1], P_[1]), veccpy!(P[2], P_[2]), veccpy!(P[3], P_[3])

      A = (
        alloc_mem_sf!(iwork, n + 1, $(config[:mem_A])),
        alloc_mem_sf!(iwork, Annz, $(config[:mem_A])),
        alloc_mem_sf!(fwork, Annz, $(config[:mem_A])),
      ) # alloc A
      if thread_idx == 1
        veccpy!(A[1], A_[1]), veccpy!(A[2], A_[2]), veccpy!(A[3], A_[3])
        sptranspose!(AT..., A...)
      end
      ##################################################################################################

      # create an identity matrix ##################################################
      I = (
        alloc_mem_sf!(iwork, m + 1, $(config[:mem_I])),
        alloc_mem_sf!(iwork, m, $(config[:mem_I])),
        alloc_mem_sf!(fwork, m, $(config[:mem_I])),
      ) # alloc I
      if thread_idx == 1
        for i in 1:m
          @cinbounds I[1][i], I[2][i], I[3][i] = i, i, -1.0f0 / rho0
        end
        @cinbounds I[1][m+1] = m + 1
      end

      # allocate the temporary matrix for combined hessian matrix ###################
      T_upper = (
        alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_T])),
        alloc_mem_sf!(iwork, Pnnz + Annz, $(config[:mem_T])),
        alloc_mem_sf!(fwork, Pnnz + Annz, $(config[:mem_T])),
      ) # alloc T_upper
      T_lower = (
        alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_T])),
        alloc_mem_sf!(iwork, Annz + m, $(config[:mem_T])),
        alloc_mem_sf!(fwork, Annz + m, $(config[:mem_T])),
      ) # alloc T_lower

      if thread_idx == 1
        sphcat!(T_upper..., P..., AT...)
        T_upper = trim_spmat(T_upper...)
        sphcat!(T_lower..., A..., I...)
        T_lower = trim_spmat(T_lower...)
        spvcat!(H..., n, T_upper..., T_lower...)
        H = trim_spmat(H...)
      end

      imem_fast = max(imem_fast, fast_buffer_used(iwork))
      fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
      imem_slow = max(imem_slow, slow_buffer_used(iwork))
      fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
      free_mem_sf!(iwork, n + m + 1, $(config[:mem_T]))
      free_mem_sf!(iwork, Annz + m, $(config[:mem_T]))
      free_mem_sf!(fwork, Annz + m, $(config[:mem_T])) # free T_lower
      free_mem_sf!(iwork, n + m + 1, $(config[:mem_T]))
      free_mem_sf!(iwork, Pnnz + Annz, $(config[:mem_T]))
      free_mem_sf!(fwork, Pnnz + Annz, $(config[:mem_T])) # free T_upper

      free_mem_sf!(iwork, m + 1, $(config[:mem_I]))
      free_mem_sf!(iwork, m, $(config[:mem_I]))
      free_mem_sf!(fwork, m, $(config[:mem_I])) # free I
      free_mem_sf!(fwork, Annz, $(config[:mem_A]))
      free_mem_sf!(iwork, Annz, $(config[:mem_A]))
      free_mem_sf!(iwork, m + 1, $(config[:mem_A])) # free A
      free_mem_sf!(fwork, Pnnz + n, $(config[:mem_P]))
      free_mem_sf!(iwork, Pnnz + n, $(config[:mem_P]))
      free_mem_sf!(iwork, n + 1, $(config[:mem_P])) # free P
      free_mem_sf!(fwork, Annz, $(config[:mem_A]))
      free_mem_sf!(iwork, Annz, $(config[:mem_A]))
      free_mem_sf!(iwork, n + 1, $(config[:mem_A])) # free AT

      # compute the permutation and ordering ########################################


      if thread_idx == 1
        H = trim_spmat(H...)
      end
      sync_threads()
      Hnnz = H[1][end] - 1
      if $(config[:use_amd] == 1)
        perm = alloc_mem_sf!(iwork, n + m, $(config[:mem_perm]))
        n_bits = AMDPKG.compute_n_bits(n + m)
        ordering_iwork = alloc_mem_sf!(iwork, 2 * (n + m + n_bits), 0)
        if thread_idx == 1
          AMDPKG.find_ordering(perm, H[1], H[2], ordering_iwork)
        end
        imem_fast = max(imem_fast, fast_buffer_used(iwork))
        fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
        imem_slow = max(imem_slow, slow_buffer_used(iwork))
        fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
        free_mem_sf!(iwork, 2 * (n + m + n_bits), 0)
        iperm = alloc_mem_sf!(iwork, n + m, $(config[:mem_perm]))
        iperm_iwork = alloc_mem_sf!(iwork, n + m, $(config[:mem_perm_work]))
        if thread_idx == 1
          AMDPKG.mergeargsort!(iperm, perm, 1, length(perm), iperm_iwork)
        end
        imem_fast = max(imem_fast, fast_buffer_used(iwork))
        fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
        imem_slow = max(imem_slow, slow_buffer_used(iwork))
        fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
        free_mem_sf!(iwork, n + m, $(config[:mem_perm_work]))
        H_perm = (
          alloc_mem_sf!(iwork, n + m + 1, $(config[:mem_H_perm])),
          alloc_mem_sf!(iwork, Hnnz, $(config[:mem_H_perm])),
          alloc_mem_sf!(fwork, Hnnz, $(config[:mem_H_perm])),
        )
        permute_iwork = alloc_mem_sf!(iwork, 2 * (n + m), $(config[:mem_H_perm_work]))
        if thread_idx == 1
          AMDPKG.permute_mat(H_perm, H, perm, iperm, permute_iwork)
        end
        imem_fast = max(imem_fast, fast_buffer_used(iwork))
        fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
        imem_slow = max(imem_slow, slow_buffer_used(iwork))
        fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
        free_mem_sf!(iwork, 2 * (n + m), $(config[:mem_H_perm_work]))
        H = H_perm
      end

      #cuprint_matrix(H)
      if thread_idx == 1
        sptriu!(H...)
      end

      # allocate and compute the LDLT factorization ################################

      Lnz = alloc_mem_sf!(iwork, n + m, $(config[:mem_Lnz]))
      info = alloc_mem_sf!(iwork, 1, 1)
      etree = alloc_mem_sf!(iwork, n + m, $(config[:mem_etree]))
      etree_iwork = alloc_mem_sf!(iwork, n + m, $(config[:mem_etree_iwork])) # alloc etree_iwork

      if thread_idx == 1
        LDLT_etree!(n + m, H[1:2]..., etree_iwork, Lnz, info, etree)
      end
      imem_fast = max(imem_fast, fast_buffer_used(iwork))
      fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
      imem_slow = max(imem_slow, slow_buffer_used(iwork))
      fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
      free_mem_sf!(iwork, n + m, $(config[:mem_etree_iwork])) # free etree_iwork

      if thread_idx == 1 && info[1] <= 0 #@cuassert info[1] > 0
        vecfill!(sol, NaN)
        @cuprintf("Etree failed on idx = %d\n", Int32(idx))
        return
      end
      sync_threads()
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

      if thread_idx == 1
        LDLT_factor!(n + m, H, L, D, Dinv, info, Lnz, etree, ldlt_iwork, ldlt_fwork, false)
      end
      imem_fast = max(imem_fast, fast_buffer_used(iwork))
      fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
      imem_slow = max(imem_slow, slow_buffer_used(iwork))
      fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
      free_mem_sf!(iwork, 4 * (n + m), $(config[:mem_ldlt_iwork])) # free ldlt_iwork
      free_mem_sf!(fwork, 2 * (n + m), $(config[:mem_ldlt_fwork])) # free ldlt_fwork
      if thread_idx == 1 && info[1] <= 0 # @cuassert info[1] > 0
        vecfill!(sol, NaN)
        @cuprintf("factor failed on idx = %d\n", Int32(idx))
        return
      end

      # allocate the ADMM variables ###############################################
      temp = alloc_mem_sf!(fwork, n + m, $(config[:mem_temp]))
      temp2 = alloc_mem_sf!(fwork, n + m, $(config[:mem_temp]))
      x = alloc_mem_sf!(fwork, n, $(config[:mem_x]))
      v = alloc_mem_sf!(fwork, m, $(config[:mem_v]))
      z = alloc_mem_sf!(fwork, m, $(config[:mem_z]))
      y = alloc_mem_sf!(fwork, m, $(config[:mem_y]))

      sync_threads()

      # solve in a loop for a fixed number of iterations ##########################
      k = 0

      n_thread_sidx = ceil(Int32, n / n_threads) * (threadIdx().x - 1) + 1
      m_thread_sidx = ceil(Int32, m / n_threads) * (threadIdx().x - 1) + 1
      nm_thread_sidx = ceil(Int32, (n + m) / n_threads) * (threadIdx().x - 1) + 1
      n_thread_eidx = min(ceil(Int32, n / n_threads) * threadIdx().x, n)
      m_thread_eidx = min(ceil(Int32, m / n_threads) * threadIdx().x, m)
      nm_thread_eidx = min(ceil(Int32, (n + m) / n_threads) * threadIdx().x, n + m)

      for it in Int32(1):Int32(iters)
        k += 1
        admm_set_rhs_top!(view(temp, 1:n), sig, x, q, n_thread_sidx, n_thread_eidx)
        admm_set_rhs_bot!(view(temp, n+1:n+m), z, y, rho0, m_thread_sidx, m_thread_eidx)

        sync_threads()

        # solve the problem
        if $(config[:use_amd] == 1)
          vecpermute!(temp2, temp, perm, nm_thread_sidx, nm_thread_eidx)
          sync_threads()
          LDLT_solve!(n + m, L..., Dinv, temp2, n_threads)
          sync_threads()
          vecpermute!(temp, temp2, iperm, nm_thread_sidx, nm_thread_eidx)
        else
          LDLT_solve!(n + m, L..., Dinv, temp, n_threads)
        end
        sync_threads()

        veccpy!(x, view(temp, 1:n), n_thread_sidx, n_thread_eidx)
        veccpy!(v, view(temp, n+1:n+m), m_thread_sidx, m_thread_eidx)
        sync_threads()
        admm_update_z!(z, v, y, rho0, n_thread_sidx, n_thread_eidx)
        admm_update_y!(y, z, l, u, rho0, m_thread_sidx, m_thread_eidx)
        sync_threads()
      end

      # copy the result into the solution vector ###################################
      veccpy!(view(sol, 1:n), x, n_thread_sidx, n_thread_eidx)
      veccpy!(view(sol, n+1:n+m), y, m_thread_sidx, m_thread_eidx)
      sync_threads()

      # show the memory usage ######################################################
      imem_fast = max(imem_fast, fast_buffer_used(iwork))
      fmem_fast = max(fmem_fast, fast_buffer_used(fwork))
      imem_slow = max(imem_slow, slow_buffer_used(iwork))
      fmem_slow = max(fmem_slow, slow_buffer_used(fwork))
      if thread_idx == 1
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
