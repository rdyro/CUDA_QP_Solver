using CUDA

speye(n) = sparse(I, n, n)

####################################################################################################

USE_DUMB = 1
USE_SCORES = 2

####################################################################################################

const BIT_SIZE = 31

function compute_n_bits(x)
  return div(x, BIT_SIZE) + 1
end

@inline function element2bit_idx(idx)
  elem_idx = div(idx, BIT_SIZE) + 1
  bit_idx = mod(idx, BIT_SIZE)
  return (elem_idx, bit_idx)
end

@inline function check_and_set_neighbor!(set_work, idx; mutate=true)
  elem_idx, bit_idx = element2bit_idx(idx)
  @cinbounds exists = ((set_work[elem_idx] & (1 << bit_idx)) != 0)
  (mutate) && (set_work[elem_idx] |= (1 << bit_idx))
  return exists ? 0 : 1
end

@inline function is_in_set(set_work, idx)
  elem_idx, bit_idx = element2bit_idx(idx)
  @cinbounds exists = ((set_work[elem_idx] & (1 << bit_idx)) != 0)
  return exists
end

@inline function set_in_set!(set_work, idx)
  elem_idx, bit_idx = element2bit_idx(idx)
  @cinbounds set_work[elem_idx] |= UInt32(1 << bit_idx)
  return
end

@inline function clear_in_set!(set_work, idx)
  elem_idx, bit_idx = element2bit_idx(idx)
  @cinbounds set_work[elem_idx] &= ~T(1 << bit_idx)
  return
end

####################################################################################################

@inline function get_neighbors(Ap, Ai, i)
  @cinbounds return @view Ai[Ap[i]:Ap[i+1]-1]
end

####################################################################################################


function queue_push!(queue::Vec{T}, queue_set::Vec{T}, queue_length::T, el::T)::T where {T}
  (is_in_set(queue_set, el)) && (return queue_length)
  @cinbounds queue[queue_length+1] = el
  set_in_set!(queue_set, el)
  return queue_length + 1
end

function queue_pop!(queue::Vec{T}, queue_set::Vec{T}, queue_length::T)::T where {T}
  @cinbounds el = queue[queue_length]
  clear_in_set!(queue_set, el)
  return el
end

####################################################################################################

function vertex_weight_3depth(Ap, Ai, is_enode_mask, set_work, v)
  weight = 0
  n = length(Ap) - 1
  n_bits = div(n, BIT_SIZE) + 1
  for i in 1:n_bits
    @cinbounds set_work[i] = 0
  end
  for n1 in get_neighbors(Ap, Ai, v)
    if is_in_set(is_enode_mask, n1) && !is_in_set(set_work, n1)
      for n2 in get_neighbors(Ap, Ai, n1)
        if is_in_set(is_enode_mask, n2) && !is_in_set(set_work, n2)
          for n3 in get_neighbors(Ap, Ai, n2)
            weight += check_and_set_neighbor!(set_work, n3)
          end
        else
          weight += check_and_set_neighbor!(set_work, n2)
        end
      end
    else
      weight += check_and_set_neighbor!(set_work, n1)
    end
  end
  return weight
end

function vertex_weight_5depth(
  Ap::Vec{T},
  Ai::Vec{T},
  is_enode_mask::Vec{T},
  set_work::Vec{T},
  v::Int,
)::T where {T}
  weight = 0
  n = length(Ap) - 1
  n_bits = div(n, BIT_SIZE) + 1
  @csimd for i in 1:n_bits
    @cinbounds set_work[i] = 0
  end
  for n1 in get_neighbors(Ap, Ai, v)
    if is_in_set(is_enode_mask, n1) && !is_in_set(set_work, n1)
      for n2 in get_neighbors(Ap, Ai, n1)
        if is_in_set(is_enode_mask, n2) && !is_in_set(set_work, n2)
          for n3 in get_neighbors(Ap, Ai, n2)
            if is_in_set(is_enode_mask, n3) && !is_in_set(set_work, n3)
              for n4 in get_neighbors(Ap, Ai, n3)
                if is_in_set(is_enode_mask, n4) && !is_in_set(set_work, n4)
                  for n5 in get_neighbors(Ap, Ai, n4)
                    weight += check_and_set_neighbor!(set_work, n5)
                  end
                else
                  weight += check_and_set_neighbor!(set_work, n4)
                end
              end
            else
              weight += check_and_set_neighbor!(set_work, n3)
            end
          end
        else
          weight += check_and_set_neighbor!(set_work, n2)
        end
      end
    else
      weight += check_and_set_neighbor!(set_work, n1)
    end
  end
  return weight
end

function vertex_weight_amd(
  Ap::Vec{T},
  Ai::Vec{T},
  weights::Vec{T},
  is_enode_mask::Vec{T},
  set_work::Vec{T},
  v::Int,
)::T where {T}
  weight = 0
  n = length(Ap) - 1
  n_bits = div(n, BIT_SIZE) + 1
  @csimd for i in 1:n_bits
    @cinbounds set_work[i] = 0
  end
  for n1 in get_neighbors(Ap, Ai, v)
    if is_in_set(is_enode_mask, n1)
      weight += weights[n1]
      for n2 in get_neighbors(Ap, Ai, n1)
        set_in_set!(set_work, n2)
        #(is_in_set(set_work, n2)) && (weight -= is_in_set(is_enode_mask, n2) ? weights[n2] : 1)
        (is_in_set(set_work, n2)) && (weight -= 1)
      end
    else
      weight += check_and_set_neighbor!(set_work, n1)
    end
  end
  return weight
end

function vertex_weight_exact(
  Ap::Vec{T},
  Ai::Vec{T},
  is_enode_set::Vec{T},
  neighbor_set::Vec{T},
  queue::Vec{T},
  queue_set::Vec{T},
  v::Int,
)::T where {T}
  weight = 0
  n = length(Ap) - 1
  n_bits = div(n, BIT_SIZE) + 1
  @csimd for i in 1:n_bits
    @cinbounds neighbor_set[i] = 0
    @cinbounds queue_set[i] = 0
  end
  queue_len = 0
  for n in get_neighbors(Ap, Ai, v)
    queue_len = queue_push!(queue, queue_set, queue_len, n)
  end
  while queue_len > 0
    n = queue_pop!(queue, queue_set, queue_len)
    queue_len -= 1
    new_neigh = check_and_set_neighbor!(neighbor_set, n)
    (!is_in_set(is_enode_set, n)) && (weight += new_neigh)
    if is_in_set(is_enode_set, n)
      for n1 in get_neighbors(Ap, Ai, n)
        if !is_in_set(neighbor_set, n1)
          queue_len = queue_push!(queue, queue_set, queue_len, n1)
        end
      end
    else
    end
  end
  return weight
end

####################################################################################################

function compute_fillin(Ap::Vec{T}, Ai::Vec{T}, perm::Vec{T}, iwork::Vec{T}) where {T}
  @cinbounds begin
    n = length(Ap) - 1
    n_bits = div(n, BIT_SIZE) + 1
    @cinbounds queue = @view iwork[1:n]
    @cinbounds queue_set = @view iwork[n+1:n+n_bits]
    @cinbounds is_enode_set = @view iwork[n+n_bits+1:n+2*n_bits]
    @cinbounds neighbor_set = @view iwork[n+2*n_bits+1:n+3*n_bits]
    @csimd for i in 1:n_bits
      @cinbounds is_enode_set[i] = 0
    end
  end

  @cinbounds begin
    nnz = 0
    for n in perm
      nnz += vertex_weight_exact(Ap, Ai, is_enode_set, neighbor_set, queue, queue_set, n)
      set_in_set!(is_enode_set, n)
    end
  end
  return nnz
end

####################################################################################################


function find_ordering(
  perm::Vec{T},
  Ap::Vec{T},
  Ai::Vec{T},
  iwork::Vec{T};
  #method::Symbol=:depth3,
)::Nothing where {T}
  #@assert method in (:depth3, :depth5, :amd, :frozen)

  # initialize
  begin
    n = length(Ap) - 1
    n_bits = div(n, BIT_SIZE) + 1
    # map working memory
    k = 0
    to_consider, k = (@view iwork[k+1:k+n]), k + n
    neighbor_set, k = (@view iwork[k+1:k+n_bits]), k + n_bits
    is_enode_set, k = (@view iwork[k+1:k+n_bits]), k + n_bits
    weights, k = (@view iwork[k+1:k+n]), k + n

    @cinbounds for i in 1:n
      weights[i] = length(get_neighbors(Ap, Ai, i))
      to_consider[i] = i
    end
    @cinbounds for i in 1:n_bits
      is_enode_set[i] = 0
    end
  end
  # build up the heuristic-driven permutation
  begin
    @cinbounds for i in 1:n
      node_idx, j_found = to_consider[1], 1
      @cinbounds for j in 2:(n-i+1)
        node = to_consider[j]
        if weights[node] < weights[node_idx]
          node_idx = node
          j_found = j
        end
      end

      # update AMD weights 
      set_in_set!(is_enode_set, node_idx)
      for n in get_neighbors(Ap, Ai, node_idx)
        @cinbounds weights[n] = vertex_weight_3depth(Ap, Ai, is_enode_set, neighbor_set, n)
        #if method == :depth3
        #  @inbounds weights[n] = vertex_weight_3depth(Ap, Ai, is_enode_set, neighbor_set, n)
        #elseif method == :depth5
        #  @inbounds weights[n] = vertex_weight_5depth(Ap, Ai, is_enode_set, neighbor_set, n)
        #elseif method == :amd
        #  @inbounds weights[n] = vertex_weight_amd(Ap, Ai, weights, is_enode_set, neighbor_set, n)
        #end # else froze, do nothing
      end
      @cinbounds to_consider[j_found] = to_consider[n-i+1]
      @cinbounds to_consider[n-i+1] = node_idx

      # mark min_idx handled
      @cinbounds perm[i] = node_idx
    end
  end
  return
end
