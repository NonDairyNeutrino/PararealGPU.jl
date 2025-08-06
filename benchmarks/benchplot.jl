using DelimitedFiles, Plots

# single_time_matrix = replace(readdlm("single_time_matrix.tsv"), -1.0 => missing)
# gpu_time_matrix    = replace(readdlm("gpu_time_matrix.tsv"), -1.0 => missing)
distributed_time_matrix = readdlm("distributed_time_matrix.tsv")

coarse_vector      = collect(3:14)
fine_vector        = collect(3:14)
coarse_fine_matrix = Iterators.product(coarse_vector, fine_vector) |> collect

bench_plt_coarse = plot(
    coarse_vector,
    gpu_time_matrix .|> log10,
    # title  = "Single Threaded",
    xlabel = "log2 coarse discretization",
    ylabel = "log10 time [s]",
    labels = string.("F = 2^", fine_vector) |> permutedims,
    xticks = coarse_vector,
    # dpi    = 200,
    # size   = (3 * 200, 2 * 200)
)
# savefig(plt_single_coarse, "bench_single_coarse.png")

bench_plt_fine = plot(
    fine_vector, 
    gpu_time_matrix .|> log10 |> permutedims,
    # title = "Single Threaded",
    xlabel = "log2 fine discretization",
    # ylabel = "log10 time (s)",
    labels = string.("C = 2^", coarse_vector) |> permutedims,
    # xscale = :log2,
    xticks = fine_vector,
    yticks = []
    # dpi    = 200,
    # size   = (3 * 200, 2 * 200)
)
# savefig(plt_single_fine, "bench_single_fine.png")

bench_plt = plot(
    bench_plt_coarse, 
    bench_plt_fine,
    suptitle = "Single GPU",
    link = :y, 
    size   = (3 * 200, 2 * 200)
)
display(bench_plt)
savefig("bench_gpu.png")
