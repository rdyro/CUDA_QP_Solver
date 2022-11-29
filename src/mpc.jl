using LinearAlgebra, SparseArrays

################################################################################
import SparseArrays.sparse
function sparse(A::AbstractArray{T}, keepzeros) where {T<:Number}
  if keepzeros == false
    return sparse(A)
  else
    out = sparse(ones(T, size(A[:, :])))
    out.nzval .= A[:]
    return out
  end
end

function construct_Ab(f, fx, fu)
  @assert length(f) == length(fx) == length(fu) > 0
  N, xdim, udim = length(f), size(fx[1], 2), size(fu[1], 2)
  fx_, fu_ = sparse.(fx, true), sparse.(fu, true)
  Ax = add(
    sparse(-I, N * xdim, N * xdim),
    hcat(
      vcat(spzeros(xdim, (N - 1) * xdim), blockdiag(fx_[2:end]...)),
      spzeros(xdim * N, xdim),
    ),
    true,
  )
  Au = blockdiag(fu_...)
  b = -vcat(f...)

  return Ax, Au, b
end

function construct_Ab!(Ap, Ai, Ax, b, f, fx, fu)
  N, xdim, udim = size(fx, 3), size(fx, 2), size(fu, 2)
  #nnz = N * (xdim + xdim * xdim + xdim * udim)
  Ap[1] = 1
  for i in 1:N*udim
    inc = xdim
    Ap[i+1] = Ap[i] + inc
  end
  for i in N*udim+1:N*udim+(N-1)*xdim
    inc = 1 + xdim
    Ap[i+1] = Ap[i] + inc
  end
  for i in N*udim+(N-1)*xdim+1:N*udim+N*xdim
    inc = 1
    Ap[i+1] = Ap[i] + inc
  end

  l = 0
  for i in 1:N
    for j in 1:udim
      for k in 1:xdim
        l += 1
        Ai[l] = (i - 1) * xdim + k
        Ax[l] = fu[k, j, i]
      end
    end
  end
  for i in 1:N-1
    for j in 1:xdim
      l += 1
      Ai[l] = (i - 1) * xdim + j
      Ax[l] = -1.0
      for k in 1:xdim
        l += 1
        Ai[l] = (i - 1) * xdim + xdim + k
        Ax[l] = fx[k, j, i+1]
      end
    end
  end
  for j in 1:xdim
    l += 1
    Ai[l] = (N - 1) * xdim + j
    Ax[l] = -1.0
  end

  for i in 1:N
    for j in 1:xdim
      b[xdim*(i-1)+j] = -f[j, i]
    end
  end
  return
end
