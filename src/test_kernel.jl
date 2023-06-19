function test_kernel(data, n)
  #(threadIdx().x != 1 || blockIdx().x != 1) && (return)

  iwork = CUDA.@cuStaticSharedMem(Int32, 10000)
  fwork = CUDA.@cuStaticSharedMem(Float32, 10000)
  a, fwork = view_mem(fwork, len32(data))
  b, fwork = view_mem(fwork, len32(data))
  c, fwork = view_mem(fwork, len32(data))
  d, fwork = view_mem(fwork, len32(data))
  e, fwork = view_mem(fwork, n)

  for i in 1:len32(data)
    #a[i] = data[i]
    b[i] = 1.0 * i
    c[i] = 1.0 * i
    d[i] = 1.0 * 2
  end
  for i in 1:len32(data)
    #data[i] = a[i] * b[i]
    data[i] = b[i] * b[i] - c[i]
  end
  data[end] = len32(fwork)
  return nothing
end

const S = 1000

@inline function has_empty(Ap)
  any_same = false
  for i in 1:(len32(Ap) - Int32(1))
    any_same |= (Ap[i] == Ap[i+1])
  end
  return any_same
end
