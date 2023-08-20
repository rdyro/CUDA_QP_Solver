const LDLT_UNKNOWN = -1
const LDLT_USED = 1
const LDLT_UNUSED = 0

################################################################################
function LDLT_etree!(n, Ap, Ai, iwork, Lnz, info, etree)
  for i in 1:n  #= @inbounds =#
    # zero out Lnz and work. Set all etree values to unknown
    iwork[i] = 0
    Lnz[i] = 0
    etree[i] = LDLT_UNKNOWN

    # Abort if A doesn't have at least one entry one entry in every column
    if Ap[i] == Ap[i+1]
      info[1] = -2
      return
    end
  end

  for j in 1:n
    iwork[j] = j
    for p in Ap[j]:Ap[j+1]-1
      i = Ai[p]
      if i > j
        info[1] = -1
        return
      end
      while iwork[i] != j
        if etree[i] == LDLT_UNKNOWN
          etree[i] = j
        end
        Lnz[i] += 1        # nonzeros in this column
        iwork[i] = j
        i = etree[i]
      end
    end
  end

  info[1] = 0
  for i in 1:n
    info[1] += Lnz[i]
  end
  return
end

function LDLT_factor!(
  n,
  A,
  L,
  D,
  Dinv,
  info,
  Lnz,
  etree,
  iwork,
  fwork,
  logicalFactor,
)
  Ap, Ai, Ax = A
  Lp, Li, Lx = L
  positiveValuesInD = 0

  #partition working memory into pieces
  yMarkers = view(iwork, 1:n)
  yIdx = view(iwork, n+1:2*n)
  elimBuffer = view(iwork, 2*n+1:3*n)
  LNextSpaceInCol = view(iwork, 3*n+1:4*n)
  yVals = fwork

  Lp[1] = 1 #first column starts at index one / Julia is 1 indexed
  @cinbounds for i in 1:n
    #compute L column indices
    Lp[i+1] = Lp[i] + Lnz[i]   #cumsum, total at the end

    # set all Yidx to be 'unused' initially in each column of L, the next
    # available space to start is just the first space in the column
    yMarkers[i] = LDLT_UNUSED
    yVals[i] = 0f0
    D[i] = 0f0
    LNextSpaceInCol[i] = Lp[i]
  end

  if !logicalFactor
    # First element of the diagonal D.
    D[1] = Ax[1]
    if D[1] == 0f0
      #println("Singular exception in factor")
      info[1] = -1
      return
    end
    if D[1] > 0f0
      positiveValuesInD += 1
    end
    Dinv[1] = 1f0 / D[1]
  end

  #Start from 1 here. The upper LH corner is trivially 0 in L b/c we are only
  # computing the subdiagonal elements
  @cinbounds for k in 2:n
    #NB : For each k, we compute a solution to
    #y = L(1:k), 1:k)\b, where b is the kth
    #column of A that sits above the diagonal.
    #The solution y is then the kth row of L,
    #with an implied '1' at the diagonal entry.

    #number of nonzeros in this row of L
    nnzY = 0  #number of elements in this row

    #This loop determines where nonzeros
    #will go in the kth row of L, but doesn't
    #compute the actual values
    @cinbounds for i in Ap[k]:(Ap[k+1]-1)
      bidx = Ai[i]   # we are working on this element of b
      #Initialize D[k] as the element of this column
      #corresponding to the diagonal place.  Don't use
      #this element as part of the elimination step
      #that computes the k^th row of L
      if (bidx == k)
        D[k] = Ax[i]
        continue
      end
      yVals[bidx] = Ax[i]   # initialise y(bidx) = b(bidx)

      # use the forward elimination tree to figure
      # out which elements must be eliminated after
      # this element of b
      nextIdx = bidx
      if yMarkers[nextIdx] == LDLT_UNUSED  #this y term not already visited
        yMarkers[nextIdx] = LDLT_USED     #I touched this one
        elimBuffer[1] = nextIdx  # It goes at the start of the current list
        nnzE = 1         #length of unvisited elimination path from here

        nextIdx = etree[bidx]
        @cinbounds while nextIdx != LDLT_UNKNOWN && nextIdx < k
          if yMarkers[nextIdx] == LDLT_USED
            break
          end

          yMarkers[nextIdx] = LDLT_USED   #I touched this one
          #NB: Julia is 1-indexed, so I increment nnzE first here,
          #no after writing into elimBuffer as in the C version
          nnzE += 1                   #the list is one longer than before
          elimBuffer[nnzE] = nextIdx #It goes in the current list
          nextIdx = etree[nextIdx]   #one step further along tree
        end

        # now I put the buffered elimination list into
        # my current ordering in reverse order
        @cinbounds while (nnzE != 0)
          #NB: inc/dec reordered relative to C because
          #the arrays are 1 indexed
          nnzY += 1
          yIdx[nnzY] = elimBuffer[nnzE]
          nnzE -= 1
        end
      end
    end

    #This for loop places nonzeros values in the k^th row
    @cinbounds for i in nnzY:-1:1
      #which column are we working on?
      cidx = yIdx[i]

      # loop along the elements in this
      # column of L and subtract to solve to y
      tmpIdx = LNextSpaceInCol[cidx]

      #don't compute Lx for logical factorisation
      #this is not implemented in the C version
      if !logicalFactor
        yVals_cidx = yVals[cidx]
        @cinbounds for j in Lp[cidx]:(tmpIdx-1)
          yVals[Li[j]] -= Lx[j] * yVals_cidx
        end

        #Now I have the cidx^th element of y = L\b.
        #so compute the corresponding element of
        #this row of L and put it into the right place
        Lx[tmpIdx] = yVals_cidx * Dinv[cidx]

        #D[k] -= yVals[cidx]*yVals[cidx]*Dinv[cidx];
        D[k] -= yVals_cidx * Lx[tmpIdx]
      end

      #also record which row it went into
      Li[tmpIdx] = k

      LNextSpaceInCol[cidx] += 1

      #reset the yvalues and indices back to zero and LDLT_UNUSED
      #once I'm done with them
      yVals[cidx] = 0f0
      yMarkers[cidx] = LDLT_UNUSED
    end

    #Maintain a count of the positive entries
    #in D.  If we hit a zero, we can't factor
    #this matrix, so abort
    if D[k] == 0f0
      #println("Singular exception in factor")
      info[1] = -1
      return
    end
    if D[k] > 0f0
      positiveValuesInD += 1
    end

    #compute the inverse of the diagonal
    Dinv[k] = 1f0 / D[k]
  end

  #return positiveValuesInD
  info[1] = positiveValuesInD
  return
end

# Solves (L+I)x = b, with x replacing b
@inline function LDLT_Lsolve!(n, Lp, Li, Lx, x)
  for i in 1:n
    @cinbounds for j in Lp[i]:Lp[i+1]-1
      @cinbounds x[Li[j]] -= Lx[j] * x[i]
    end
  end
  return
end

# Solves (L+I)'x = b, with x replacing b
@inline function LDLT_LTsolve!(n, Lp, Li, Lx, x)
  for i in n:-1:1
    @cinbounds for j in Lp[i]:Lp[i+1]-1
      @cinbounds x[i] -= Lx[j] * x[Li[j]]
    end
  end
  return
end

# Solves Ax = b where A has given LDL factors,
# with x replacing b
@inline function LDLT_solve!(n, Lp, Li, Lx, Dinv, b)
  LDLT_Lsolve!(n, Lp, Li, Lx, b)
  @csimd for i in 1:n
    @cinbounds b[i] = b[i] * Dinv[i]
  end
  LDLT_LTsolve!(n, Lp, Li, Lx, b)
  return
end
####################################################################################################

# Solves (L+I)x = b, with x replacing b
@inline function LDLT_Lsolve!(n, Lp, Li, Lx, x, n_threads)
  for i in 1:n
    npt = ceil(Int32, (Lp[i+1] - Lp[i]) / n_threads) # n per thread
    s, e = npt * (threadIdx().x - 1) + Lp[i], min(npt * threadIdx().x, Lp[i+1]-1)
    @cinbounds for j in s:e
      @cinbounds x[Li[j]] -= Lx[j] * x[i]
    end
    sync_threads()
  end
  return
end

# Solves (L+I)'x = b, with x replacing b
@inline function LDLT_LTsolve!(n, Lp, Li, Lx, x, n_threads)
  for i in n:-1:1
    npt = ceil(Int32, (Lp[i+1] - Lp[i]) / n_threads) # n per thread
    s, e = npt * (threadIdx().x - 1) + Lp[i], min(npt * threadIdx().x, Lp[i+1]-1)
    @cinbounds for j in s:e
      @cinbounds x[i] -= Lx[j] * x[Li[j]]
    end 
    sync_threads()
  end
  return
end

# Solves Ax = b where A has given LDL factors,
# with x replacing b
@inline function LDLT_solve!(n, Lp, Li, Lx, Dinv, b, n_threads)
  thread_idx = threadIdx().x

  #LDLT_Lsolve!(n, Lp, Li, Lx, b, n_threads)
  if thread_idx == 1
    LDLT_Lsolve!(n, Lp, Li, Lx, b)
  end
  sync_threads()

  npt = ceil(Int32, n / n_threads) # n per thread
  s, e = npt * (thread_idx - 1) + 1, min(npt * thread_idx, n)
  for i in s:e
    @cinbounds b[i] = b[i] * Dinv[i]
  end
  sync_threads()

  #LDLT_LTsolve!(n, Lp, Li, Lx, b, n_threads)
  if thread_idx ==  1
    LDLT_LTsolve!(n, Lp, Li, Lx, b)
  end
  sync_threads()
  return
end