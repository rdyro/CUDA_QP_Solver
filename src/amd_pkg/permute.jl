unpack_sparse_mat(A::SparseMatrixCSC) = (A.colptr, A.rowval, A.nzval)

function permute_mat(P::SparseTuple, A::SparseTuple, perm::Vec, iperm::Vec, iwork::Vec)::Nothing
  # make memory arrays from iwork ############################
  (Ap, Ai, Ax), (Pp, Pi, Px) = A, P
  sortwork = view(iwork, 1:length(perm))
  colperm = view(iwork, length(perm)+1:2*length(perm))

  # create the inverse permutation ###########################
  # mergeargsort!(iperm, perm, 1, length(perm), sortwork)

  n = size(Ap, 1) - 1
  Pp[1] = 1
  for i in 1:n
    idx = perm[i]
    Pp[i+1] = Ap[idx+1] - Ap[idx] + Pp[i]
  end
  #@assert Pp[end] == Ap[end]

  k = 0
  for i in 1:n
    # copy the unordered column ##############################
    for j in Ap[perm[i]]:Ap[perm[i]+1]-1
      Pi[k+j-Ap[perm[i]]+1] = iperm[Ai[j]]
    end
    k += Ap[perm[i]+1] - Ap[perm[i]]

    # order the column #######################################
    s, e = Pp[i], Pp[i+1] - 1
    n = e - s + 1
    mergeargsort!(view(colperm, 1:n), view(Pi, s:e), 1, n, sortwork)

    #arg_idx = sortperm(Pi[Pp[i]:Pp[i+1]-1])
    #Pi[Pp[i]:Pp[i+1]-1] .= Pi[Pp[i]:Pp[i+1]-1][arg_idx]
    #Px[Pp[i]:Pp[i+1]-1] .= Px[Pp[i]:Pp[i+1]-1][arg_idx]

    for j in 1:n
      Pi[s+j-1] = iperm[Ai[Ap[perm[i]] + colperm[j] - 1]]
      Px[s+j-1] = Ax[Ap[perm[i]] + colperm[j] - 1]
    end
  end
  return
end


function test_permute_matrix()
  for _ in 1:100
    n = rand(100:300)
    A = sprand(n, n, 0.1)
    A = A + A' + 1e-5 * I

    perm = randperm(size(A, 1))
    iwork = zeros(Int, 3 * size(A, 1))

    P = copy.(unpack_sparse_mat(A))
    @time permute_mat(P, unpack_sparse_mat(A), perm, iwork)
    A_perm = SparseMatrixCSC(size(A)..., P...)

    A_perm_ = A[perm, :][:, perm]
    @assert norm(A_perm - A_perm_) < 1e-9
  end
end