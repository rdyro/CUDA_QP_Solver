# sort #############################################################################################

function merge!(a, lo, mi, hi, iwork)
  i, j, k = lo, mi + 1, lo
  @cinbounds while i <= mi && j <= hi
    if a[i] > a[j]
      iwork[k], j, k = a[j], j + 1, k + 1
    else
      iwork[k], i, k = a[i], i + 1, k + 1
    end
  end
  @cinbounds while i <= mi  #= @inbounds =#
    iwork[k], i, k = a[i], i + 1, k + 1
  end
  @cinbounds while j <= hi  #= @inbounds =#
    iwork[k], j, k = a[j], j + 1, k + 1
  end
  return
end

function mergesort!(a, lo, hi, iwork)
  nb = hi - lo
  (nb < 1) && (return)
  if nb == 2
    if a[lo] > a[hi]
      a[lo], a[hi] = a[hi], a[lo]
    end
  end

  step, odd_run = 1, true
  while step <= nb  #= @inbounds =#
    for i in 1:div(nb, 2 * step)+1    #= @inbounds =#
      loi = min(lo + 2 * (i - 1) * step, hi)
      mii = min(lo + 2 * (i - 1) * step + step - 1, hi)
      hii = min(lo + 2 * (i - 1) * step + 2 * step - 1, hi)
      if odd_run
        merge!(a, loi, mii, hii, iwork)
      else
        merge!(iwork, loi, mii, hii, a)
      end
    end
    odd_run = !odd_run
    step *= 2
  end
  if !odd_run
    @cinbounds for i in lo:hi
      a[i] = iwork[i]
    end
  end
  return
end


# argsort ##########################################################################################

function mergearg!(idx, a, lo, mi, hi, iwork)
  i, j, k = lo, mi + 1, lo
  @cinbounds while i <= mi && j <= hi
    if a[idx[i]] > a[idx[j]]
      iwork[k], j, k = idx[j], j + 1, k + 1
    else
      iwork[k], i, k = idx[i], i + 1, k + 1
    end
  end
  @cinbounds while i <= mi  #= @inbounds =#
    iwork[k], i, k = idx[i], i + 1, k + 1
  end
  @cinbounds while j <= hi  #= @inbounds =#
    iwork[k], j, k = idx[j], j + 1, k + 1
  end
  return
end

function mergeargsort!(idx, a, lo, hi, iwork)
  nb = hi - lo
  for i in lo:hi
    idx[i] = i
  end
  (nb < 1) && (return)
  if nb == 2 && a[lo] > a[hi]
    idx[lo], idx[hi] = hi, lo
  end

  step, odd_run = 1, true
  while step <= nb  #= @inbounds =#
    for i in 1:div(nb, 2 * step)+1    #= @inbounds =#
      loi = min(lo + 2 * (i - 1) * step, hi)
      mii = min(lo + 2 * (i - 1) * step + step - 1, hi)
      hii = min(lo + 2 * (i - 1) * step + 2 * step - 1, hi)
      if odd_run
        mergearg!(idx, a, loi, mii, hii, iwork)
      else
        mergearg!(iwork, a, loi, mii, hii, idx)
      end
    end
    odd_run = !odd_run
    step *= 2
  end
  if !odd_run
    @cinbounds for i in lo:hi
      idx[i] = iwork[i]
    end
  end
  return
end


# tests ############################################################################################

function test_sort_methods()
  for _ in 1:100
    a = rand(1:100, 10)
    iwork = zeros(Int, length(a))
    a_cpy = copy(a)

    mergesort!(a, 1, length(a), iwork)

    @assert all(sort(unique(a_cpy)) .== sort(unique(a)))

    ##################################################

    a = rand(1:100, 77)
    iwork = zeros(Int, length(a))
    a_cpy = copy(a)

    mergesort!(a, 15, 53, iwork)

    @assert all(sort(unique(a_cpy[15:53])) .== sort(unique(a[15:53])))
    @assert all(a[1:14] .== a_cpy[1:14]) && all(a[54:end] .== a_cpy[54:end])

    ##################################################

    a = rand(1:100, 77)
    iwork, idx = zeros(Int, length(a)), zeros(Int, length(a))
    a_cpy = copy(a)

    lo, hi = rand(1:15), rand(53:length(a))
    mergeargsort!(idx, a, 1, length(a), iwork)
    @assert all(sortperm(a_cpy) .== idx)

    ##################################################

    a = rand(1:100, 77)
    iwork, idx = zeros(Int, length(a)), zeros(Int, length(a))
    a_cpy = copy(a)

    lo, hi = rand(1:15), rand(53:length(a))
    mergeargsort!(idx, a, lo, hi, iwork)
    @assert all(sortperm(a_cpy[lo:hi]) .== (idx[lo:hi] .- lo .+ 1))
  end
end