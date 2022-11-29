@inline function view_mem(work, size)
  n = length(work)
  @assert size <= n
  return view(work, 1:size), view(work, size+1:n)
end

macro view_mem(work, size)
  return esc(quote
    @assert $size <= length($work)
    view($work, 1:($size)), view($work, ($size+1):(length($work)))
  end)
end
