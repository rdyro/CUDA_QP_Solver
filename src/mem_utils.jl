import Base: length

@inline trim_spmat(Ap, Ai, Ax) = (Ap, view(Ai, 1:Ap[end]-1), view(Ax, 1:Ap[end]-1))

####################################################################################################

"""Equivalent to the in-built `length` function, but returns an `Int32` instead of an `Int64`."""
@inline function len32(work::AbstractVector{T})::Int32 where {T}
  return Int32(length(work))
end

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

CUDAArrayView{T} = Union{SubArray{T,1,CuDeviceVector{T,1},Tuple{UnitRange{Int64}},true}, SubArray{T,1,CuDeviceVector{T,3},Tuple{UnitRange{Int64}},true}} where T

####################################################################################################

"""Equivalent to the in-built `length` function, but returns an `Int32` instead of an `Int64`."""
@inline function len32(work::AbstractVector{T})::Int32 where {T}
  return Int32(length(work))
end

mutable struct CPUWorkSF{T, N1, N2}
  slow_whole_buffer::AbstractArray{T, N1}
  slow_buffer::AbstractArray{T, N1}
  fast_whole_buffer::AbstractArray{T, N2}
  fast_buffer::AbstractArray{T, N2}
end

"""Make a memory block into a memory block tracking tuple."""
@inline function cpu_make_mem_sf(
  slow_buffer::AbstractArray{T, N1},
  fast_buffer::AbstractArray{T, N2},
)::CPUWorkSF{T, N1, N2} where {T, N1, N2}
  return CPUWorkSF{T, N1, N2}(
    slow_buffer,
    view(slow_buffer, 1:length(slow_buffer)),
    fast_buffer,
    view(fast_buffer, 1:length(fast_buffer)),
  )
end

"""Allocate memory from a memory block tracking tuple."""
@inline function cpu_alloc_mem_sf!(
  worksf::CPUWorkSF{T, N1, N2},
  size::Integer,
  fast::Integer,
)::Union{AbstractArray{T, N1}, AbstractArray{T, N2}} where {T, N1, N2}
  buffer = fast == 1 ? worksf.fast_buffer : worksf.slow_buffer
  @assert size <= len32(buffer)
  alloc_mem = view(buffer, 1:size)
  if fast == 1
    worksf.fast_buffer = view(buffer, (size+1):len32(buffer))
  else
    worksf.slow_buffer = view(buffer, (size+1):len32(buffer))
  end
  return alloc_mem
end

"""Free memory from a memory block tracking tuple."""
@inline function cpu_free_mem_sf!(worksf::CPUWorkSF{T, N1, N2}, size::Integer, fast::Integer)::Nothing where {T, N1, N2}
  whole_buffer = fast == 1 ? worksf.fast_whole_buffer : worksf.slow_whole_buffer
  buffer = fast == 1 ? worksf.fast_buffer : worksf.slow_buffer
  @assert size <= len32(whole_buffer) - len32(buffer)
  si, ei = len32(whole_buffer) - len32(buffer) - size + 1, len32(whole_buffer)
  if fast == 1
    worksf.fast_buffer = view(whole_buffer, si:ei)
  else
    worksf.slow_buffer = view(whole_buffer, si:ei)
  end
  return
end

@inline function fast_buffer_used(worksf::CPUWorkSF{T, N1, N2})::Int32 where {T, N1, N2} 
  return len32(worksf.fast_whole_buffer) - len32(worksf.fast_buffer)
end