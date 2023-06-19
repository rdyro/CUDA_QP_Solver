import Base: length

#const WorkT = Tuple{AbstractVector{T}, AbstractVector{T}} where T
#const Work = Array{AbstractVector{T}, 1} where T

mutable struct Work{T}
  whole_buffer::CuDeviceVector{T, 3}
  remaining_buffer::SubArray{T, 1, CuDeviceVector{T, 3}, Tuple{UnitRange{Int64}}, true}
end

@inline function length(work::Work{T})::Int32 where T
  return Int32(length(work.remaining_buffer))
end

"""Equivalent to the in-built `length` function, but returns an `Int32` instead of an `Int64`."""
@inline function len32(work::AbstractVector{T})::Int32 where T
  return Int32(length(work))
end

"""Partition a memory block into request size and rest."""
@inline function view_mem(work, size)
  @assert size <= len32(work)
  return view(work, 1:size), view(work, size+1:n)
end

"""Make a memory block into a memory block tracking tuple."""
@inline function make_mem(whole_buffer::AbstractVector{T})::Work{T} where T
  #work = Work{T}(undef, 2)
  #work[1] = whole_buffer
  #work[2] = view(whole_buffer, 1:len32(whole_buffer))
  #return work
  return Work{T}(whole_buffer, view(whole_buffer, 1:length(whole_buffer)))
end

"""Allocate memory from a memory block tracking tuple."""
@inline function alloc_mem!(work::Work{T}, size::Integer)::SubArray{T, 1, CuDeviceVector{T, 3}, Tuple{UnitRange{Int64}}, true} where T
  @assert size <= len32(work.remaining_buffer)
  alloc_mem = view(work.remaining_buffer, 1:size)
  work.remaining_buffer = view(work.remaining_buffer, (size+1):len32(work.remaining_buffer))
  return alloc_mem
end

"""Free memory from a memory block tracking tuple."""
@inline function free_mem!(work::Work{T}, size::Integer)::Nothing where T
  @assert size <= len32(work.whole_buffer) - len32(work.remaining_buffer)
  si = len32(work.whole_buffer) - len32(work.remaining_buffer) - size + 1
  ei = len32(work.whole_buffer)
  work.remaining_buffer = view(work.whole_buffer, si:ei)
  return
end