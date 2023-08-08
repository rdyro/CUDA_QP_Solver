import Base: length

@inline trim_spmat(Ap, Ai, Ax) = (Ap, view(Ai, 1:Ap[end]-1), view(Ax, 1:Ap[end]-1))

####################################################################################################

mutable struct Work{T}
  whole_buffer::CuDeviceVector{T,3}
  remaining_buffer::SubArray{T,1,CuDeviceVector{T,3},Tuple{UnitRange{Int64}},true}
end

@inline function length(work::Work{T})::Int32 where {T}
  return Int32(length(work.remaining_buffer))
end

"""Equivalent to the in-built `length` function, but returns an `Int32` instead of an `Int64`."""
@inline function len32(work::AbstractVector{T})::Int32 where {T}
  return Int32(length(work))
end

####################################################################################################

"""Partition a memory block into request size and rest."""
@inline function view_mem(work, size)
  @cuassert size <= len32(work)
  return view(work, 1:size), view(work, size+1:n)
end

####################################################################################################

"""Make a memory block into a memory block tracking tuple."""
@inline function make_mem(whole_buffer::AbstractVector{T})::Work{T} where {T}
  return Work{T}(whole_buffer, view(whole_buffer, 1:length(whole_buffer)))
end

"""Allocate memory from a memory block tracking tuple."""
@inline function alloc_mem!(
  work::Work{T},
  size::Integer,
)::SubArray{T,1,CuDeviceVector{T,3},Tuple{UnitRange{Int64}},true} where {T}
  @cuassert size <= len32(work.remaining_buffer)
  alloc_mem = view(work.remaining_buffer, 1:size)
  work.remaining_buffer = view(work.remaining_buffer, (size+1):len32(work.remaining_buffer))
  return alloc_mem
end

"""Free memory from a memory block tracking tuple."""
@inline function free_mem!(work::Work{T}, size::Integer)::Nothing where {T}
  @cuassert size <= len32(work.whole_buffer) - len32(work.remaining_buffer)
  si = len32(work.whole_buffer) - len32(work.remaining_buffer) - size + 1
  ei = len32(work.whole_buffer)
  work.remaining_buffer = view(work.whole_buffer, si:ei)
  return
end

####################################################################################################

mutable struct WorkSF{T, N1, N2}
  slow_whole_buffer::CuDeviceVector{T,N1}
  slow_buffer::SubArray{T,1,CuDeviceVector{T,N1},Tuple{UnitRange{Int64}},true}
  fast_whole_buffer::CuDeviceVector{T,N2}
  fast_buffer::SubArray{T,1,CuDeviceVector{T,N2},Tuple{UnitRange{Int64}},true}
end

"""Make a memory block into a memory block tracking tuple."""
@inline function make_mem_sf(
  slow_buffer::CuDeviceVector{T, N1},
  fast_buffer::CuDeviceVector{T, N2},
)::WorkSF{T, N1, N2} where {T, N1, N2}
  return WorkSF{T, N1, N2}(
    slow_buffer,
    view(slow_buffer, 1:length(slow_buffer)),
    fast_buffer,
    view(fast_buffer, 1:length(fast_buffer)),
  )
end

"""Allocate memory from a memory block tracking tuple."""
@inline function alloc_mem_sf!(
  worksf::WorkSF{T, N1, N2},
  size::Integer,
  fast::Integer,
)::Union{
  SubArray{T,1,CuDeviceVector{T,N1},Tuple{UnitRange{Int64}},true},
  SubArray{T,1,CuDeviceVector{T,N2},Tuple{UnitRange{Int64}},true}} where {T, N1, N2}
  buffer = fast == 1 ? worksf.fast_buffer : worksf.slow_buffer
  @cuassert size <= len32(buffer)
  alloc_mem = view(buffer, 1:size)
  if fast == 1
    worksf.fast_buffer = view(buffer, (size+1):len32(buffer))
  else
    worksf.slow_buffer = view(buffer, (size+1):len32(buffer))
  end
  return alloc_mem
end

"""Free memory from a memory block tracking tuple."""
@inline function free_mem_sf!(worksf::WorkSF{T, N1, N2}, size::Integer, fast::Integer)::Nothing where {T, N1, N2}
  whole_buffer = fast == 1 ? worksf.fast_whole_buffer : worksf.slow_whole_buffer
  buffer = fast == 1 ? worksf.fast_buffer : worksf.slow_buffer
  @cuassert size <= len32(whole_buffer) - len32(buffer)
  si, ei = len32(whole_buffer) - len32(buffer) - size + 1, len32(whole_buffer)
  if fast == 1
    worksf.fast_buffer = view(whole_buffer, si:ei)
  else
    worksf.slow_buffer = view(whole_buffer, si:ei)
  end
  return
end

@inline function fast_buffer_used(worksf::WorkSF{T, N1, N2})::Int32 where {T, N1, N2} 
  return len32(worksf.fast_whole_buffer) - len32(worksf.fast_buffer)
end

ArrayView{T} = Union{SubArray{T,1,CuDeviceVector{T,1},Tuple{UnitRange{Int64}},true}, SubArray{T,1,CuDeviceVector{T,3},Tuple{UnitRange{Int64}},true}} where T