using DelimitedFiles, Plots

single_time_matrix = readdlm("single_time_matrix.tsv")
gpu_time_matrix    = readdlm("gpu_time_matrix.tsv")
distributed_time_matrix = readdlm("distributed_time_matrix.tsv")

coarse_vector      = collect(3:14)
fine_vector        = collect(3:14)
coarse_fine_matrix = Iterators.product(coarse_vector, fine_vector) |> collect

plt_single_coarse = plot(
    first.(coarse_fine_matrix), 
    replace(single_time_matrix, -1.0 => missing) .|> log10,
    title = "Single Threaded",
    xlabel = "log2 coarse discretization",
    ylabel = "log10 time (s)",
    labels = string.("Fd = 2^", fine_vector) |> permutedims,
    # xscale = :log2,
    xticks = coarse_vector,
    dpi    = 200,
    size   = (3 * 200, 2 * 200)
)
savefig("bench_single_coarse.png", plt_single)

plt_single_fine = plot(
    (x -> x[2]).(coarse_fine_matrix), 
    replace(single_time_matrix |> permutedims, -1.0 => missing) .|> log10,
    title = "Single Threaded",
    xlabel = "log2 coarse discretization",
    ylabel = "log10 time (s)",
    labels = string.("Fd = 2^", fine_vector) |> permutedims,
    # xscale = :log2,
    xticks = coarse_vector,
    dpi    = 200,
    size   = (3 * 200, 2 * 200)
)
savefig("bench_single_coarse.png", plt_single)
