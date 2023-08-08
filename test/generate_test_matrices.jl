using JSON

include(joinpath(homedir(), "Dropbox/projects/sparse_matrix_bench_testset/matrix_market_parser.jl"))

data_file = joinpath(@__DIR__, "..", "data", "all_matrices.json")


if !isfile(data_file)
  matfiles = list_all_matrices()
  all_vals, ns = [], []
  max_n = 2000
  for matfile in matfiles
    A, header = parse_matrix_file(matfile)
    (!isreal(A) || match(r"symmetric", header) == nothing) && (continue)
    (nnz(A - A') > 0) && (A = A + A' - spdiagm(0 => diag(A)))
    (nnz(A - A') != 0) && continue
    n = size(A, 1)
    push!(ns, n)
    (n > max_n) && (continue)
    matrix = Dict(
      "colptr" => A.colptr,
      "rowval" => A.rowval,
      "nzval" => A.nzval,
      "n" => n,
      "name" => matfile,
      "header" => header,
    )
    push!(all_vals, matrix)
  end

  for i in 1:200
    n = rand(200:max_n)
    A = sprandn(n, n, 10 .^ rand(range(-4; stop=-1.5, length=1000)))
    A = A + A' + 1e-5 * I
    matrix = Dict(
      "colptr" => A.colptr,
      "rowval" => A.rowval,
      "nzval" => A.nzval,
      "n" => n,
      "name" => "random symmetric n = $n",
      "header" => "random symmetric n = $n",
    )
    push!(all_vals, matrix)
  end

  write(data_file, JSON.json(all_vals))
end