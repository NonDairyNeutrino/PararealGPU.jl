using DelimitedFiles, Plots

const bench_dir = "benchmarks/"
const single_time_matrix = replace(readdlm(bench_dir * "single_time_matrix.tsv"), -1.0 => missing)
const gpu_time_matrix    = replace(readdlm(bench_dir * "gpu_time_matrix.tsv"), -1.0 => missing)
const distributed_time_matrix = replace(readdlm(bench_dir * "distributed_time_matrix.tsv"), -1.0 => missing)
const time_matrix = distributed_time_matrix

const coarse_vector      = collect(3:14)
const fine_vector        = collect(3:14)
const coarse_fine_matrix = Iterators.product(coarse_vector, fine_vector) |> collect

function coarse_fine_comp(time_matrix :: Matrix, title :: String, filename :: String) :: Nothing
    time_matrix_log = log10.(time_matrix)

    bench_plt_coarse = plot(
        coarse_vector,
        time_matrix_log,
        xlabel = "log2 coarse discretization",
        ylabel = "log10 time [s]",
        labels = string.("F = 2^", fine_vector) |> permutedims,
        xticks = coarse_vector
    )

    bench_plt_fine = plot(
        fine_vector, 
        time_matrix_log |> permutedims,
        xlabel = "log2 fine discretization",
        labels = string.("C = 2^", coarse_vector) |> permutedims,
        xticks = fine_vector,
        yticks = []
    )

    bench_plt = plot(
        bench_plt_coarse, 
        bench_plt_fine,
        suptitle = title,
        link = :y,
    )
    display(bench_plt)
    savefig(bench_dir * filename)

    return nothing
end

function method_comp(power :: Int)
    @assert power in coarse_vector "power $power not available.  Please give $(min(coarse_vector)) < power < $(maximum(coarse_vector))"
    index = power - 2
    plt_coarse = plot(
        coarse_vector,
        [single_time_matrix[:, index] gpu_time_matrix[:, index] distributed_time_matrix[:, index]] .|> log10,
        xlabel = "log2 coarse discretization",
        ylabel = "log10 time [s]",
        title  = "Fine: $(fine_vector[index])",
        labels = ["Single" "GPU" "Distributed"],
        xticks = coarse_vector
    )

    plt_fine = plot(
        fine_vector,
        [single_time_matrix[index, :] gpu_time_matrix[index, :] distributed_time_matrix[index, :]] .|> log10,
        xlabel = "log2 fine discretization",
        # ylabel = "log10 time [s]",
        title  = "Coarse: $(coarse_vector[index])",
        labels = ["Single" "GPU" "Distributed"],
        xticks = fine_vector,
        yticks = []
    )
    
    plot(plt_coarse, plt_fine, link = :y) |> display
    # savefig(bench_dir * "method_comp.png")
end

# coarse_fine_comp(distributed_time_matrix, "Distributed", "bench_distributed.png")
bench_anim = @animate for p in coarse_vector
    method_comp(p)
end
gif(bench_anim, bench_dir * "bench.gif", fps = 1)
