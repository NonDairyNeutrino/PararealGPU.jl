#=
Measure how a combination of error and runtime changes as the coarse discretization increases
=#
using JLD2, Plots, LaTeXStrings

function calculate_error(solution #= :: Solution =#) :: Float64
    # measure energy drift with respect to the final value
    final_position = solution.positionSequence |> last |> only
    final_velocity = solution.velocitySequence |> last

    kinetic_energy   = (1//2) * sum(abs2, final_velocity)
    potential_energy = GRAVITY * RODLENGTH * (1 - cos(final_position))
    mech_energy      = kinetic_energy + potential_energy

    relative_error   = (mech_energy - TRUE_ENERGY) / TRUE_ENERGY
    return relative_error
end

"""
    calculate_efficiency(bench_seq :: NamedTuple, bench_par :: NamedTuple) :: Float64

Measure the efficiency of a parallel method by comparing its error and runtime to that of sequential.
"""
function calculate_efficiency(bench_seq, bench_par) :: Float64
    seq_sol, iters = bench_seq.value
    seq_err        = calculate_error(seq_sol)
    seq_runtime    = bench_seq.time

    par_sol, iters = bench_par.value
    par_err        = calculate_error(par_sol)
    par_runtime    = bench_par.time

    eff = (par_err * par_runtime) / (seq_err * seq_runtime)
    return eff
end

function calculate_efficiency(coarse_vector :: Vector{Int}, fine_vector :: Vector{Int}, gpu_bench) :: Matrix{Float64}
    # gpu_bench     = load(DATADIR * "bench_gpu.jld2"; nested = true)
    eff_matrix    = zeros(length(coarse_vector), length(fine_vector))

    for c in eachindex(coarse_vector)
        coarse = coarse_vector[c]
        single_bench_coarse =  load(DATADIR * "single/single_c$(coarse).jld2")["single_stored_object"]
        gpu_bench_coarse = gpu_bench["$coarse"]
        for f in eachindex(fine_vector)
            fine = fine_vector[f]
            eff_matrix[c, f] = calculate_efficiency(single_bench_coarse, gpu_bench_coarse["$fine"])
        end
    end

    return eff_matrix
end

function plot_eff(x_vector :: Vector{Int}, xlabel :: String, z_vector :: Vector{Int}, zlabel :: String, eff_matrix :: Matrix{Float64})
    #coarse plot
    plot(
        2 .^ x_vector,
        eff_matrix,
        xscale = :log2,
        yscale = :log10,
        xticks = 2 .^ x_vector,
        yticks = 10.0 .^ (-10:2:10),
        xlabel = latexstring("N_\\mathcal{$xlabel}"),
        ylabel = L"$\left| \epsilon \tau  / \epsilon_1 \tau_1 \right|$",
        labels = (f -> latexstring("N_\\mathcal{$zlabel} = 2^{$f}")).(z_vector) |> permutedims,
        leg    = :topleft,
        palette = palette([:blue, :red], 5),
        linewidth   = 2
    )

    # fine plot
    plot(
        2 .^ x_vector,
        eff_matrix,
        xscale = :log2,
        yscale = :log10,
        xticks = 2 .^ x_vector,
        yticks = 10.0 .^ (-10:2:10),
        xlabel = latexstring("N_\\mathcal{$xlabel}"),
        ylabel = L"$\left| \epsilon \tau  / \epsilon_1 \tau_1 \right|$",
        labels = (f -> latexstring("N_\\mathcal{$zlabel} = 2^{$f}")).(z_vector) |> permutedims,
        leg    = :topleft,
        palette = palette([:blue, :red], 5),
        linewidth   = 2
    )
end

function plot_eff(coarse_vector, fine_vector, bench)
    eff_matrix = calculate_efficiency(coarse_vector, fine_vector, bench)
    zeroed_em  = map(x -> isapprox(x, 0.0; atol = 10^-10) ? NaN : x, eff_matrix)
    abs_zem    = abs.(zeroed_em)

    # coarse plot
    cplot = plot(
        2 .^ coarse_vector[3:end],
        abs_zem[3:end, -2 .+ [5, 8, 9, 12]],
        xticks = 2 .^ coarse_vector[3:end],
        yticks = 10.0 .^ (-10:2:10),
        xlabel = latexstring("N_\\mathcal{C}"),
        ylabel = L"$\left| \epsilon \tau  / \epsilon_1 \tau_1 \right|$",
        labels = (f -> latexstring("N_\\mathcal{F} = 2^{$f}")).(fine_vector[-2 .+ [5, 8, 9, 12]]) |> permutedims,
        leg    = :bottomright
    )

    # fine plot
    fplot = plot(
        2 .^ fine_vector[3:end],
        permutedims(abs_zem)[3:end, -2 .+ [5, 8, 9, 12]],
        xticks = 2 .^ fine_vector[3:end],
        yticks = false,
        xlabel = latexstring("N_\\mathcal{F}"),
        labels = (f -> latexstring("N_\\mathcal{C} = 2^{$f}")).(coarse_vector[-2 .+ [5, 8, 9, 12]]) |> permutedims,
        leg    = :topleft
    )

    plot(
        cplot,
        fplot;
        xscale = :log2,
        yscale = :log10,
        layout = (1,2),
        link = :y,
        linewidth   = 2
    )
end

function plot_method_comp(coarse_vector, fine_vector, gpu_bench, dist_bench)
    cv = coarse_vector[3:end]
    fv = fine_vector[3:end]

    gpu_eff_matrix = calculate_efficiency(cv, fv, gpu_bench)
    gpu_zeroed_em  = map(x -> isapprox(x, 0.0; atol = 10^-10) ? NaN : x, gpu_eff_matrix)
    gpu_abs_zem    = abs.(gpu_zeroed_em)

    dist_eff_matrix = calculate_efficiency(cv, fv, dist_bench)
    dist_zeroed_em  = map(x -> isapprox(x, 0.0; atol = 10^-10) ? NaN : x, dist_eff_matrix)
    dist_abs_zem    = abs.(dist_zeroed_em)



    # disc_slices = [-2 .+ [6,8,10,12]]
    best_disc = 8 # 6 == 2^10

    # coarse plot
    cplot = plot(
        2 .^ cv,
        [gpu_abs_zem[:, best_disc] dist_abs_zem[:, best_disc]],
        xticks = 2 .^ cv,
        yticks = 10.0 .^ (-10:2:10),
        xlabel = latexstring("N_\\mathcal{C}"),
        ylabel = L"$\left| \epsilon \tau  / \epsilon_1 \tau_1 \right|$",
        leg    = :bottomright
    )

    # fine plot
    fplot = plot(
        2 .^ fv,
        [gpu_abs_zem[best_disc, :] dist_abs_zem[best_disc, :]],
        xticks = 2 .^ fv,
        yticks = false,
        xlabel = latexstring("N_\\mathcal{F}"),
        # labels = (f -> latexstring("N_\\mathcal{C} = 2^{$f}")).(coarse_vector[-2 .+ [5, 8, 9, 12]]) |> permutedims,
        leg    = :bottomright
    )

    plot(
        cplot,
        fplot;
        xscale = :log2,
        yscale = :log10,
        layout = (1,2),
        link = :y,
        linewidth   = 2,
        labels = ["GPU" "Dist"]
    )
end

function plot_position()
    single_plot = plot()
    jldopen(DATADIR * "bench_single.jld2") do bench
        for coarse in coarse_vector[begin+3 : end]
            domain, pos, vel = bench["$coarse"].value[1].domain
            plot!(
                single_plot,
                domain ./ PERIOD,
                pos .|> only;
                label = false #= latexstring("\\mathcal{C} = 2^{$coarse}") =#,
                leg   = :bottomleft,
                xticks = false,
                # xlabel = L"t / T",
                ylabel = L"\theta",
                annotation = [(-15, 0, "(a)")]
            )
        end
    end

    gpu_plot = plot()
    jldopen(DATADIR * "bench_gpu.jld2") do bench
        for coarse in coarse_vector[-2 .+ [6, 8 ,10, 12, 14]]
            for fine in fine_vector[-2 .+ [6, 8 ,10, 12, 14]]
                sol     = bench["$coarse/$fine"].value[1]
                domain  = sol.domain
                pos_seq = sol.positionSequence
                vel_seq = sol.velocitySequence
                plot!(
                    gpu_plot,
                    domain ./ PERIOD,
                    pos_seq .|> only,
                    label  = false #= latexstring("\\mathcal{C} = 2^{$coarse}, \\mathcal{F} = 2^{$fine}") =#,
                    leg    = :bottomleft,
                    xticks = false,
                    # xlabel = L"t / T",
                    ylabel = L"\theta",
                    annotation = [(-15, 0, "(b)")]
                )
            end
        end
    end

    dist_plot = plot()
    jldopen(DATADIR * "bench_dist.jld2") do bench
        for coarse in coarse_vector[-2 .+ [6, 8 ,10, 12, 13]]
            for fine in fine_vector[-2 .+ [6, 8 ,10, 12, 13]]
                sol     = bench["$coarse/$fine"].value[1]
                domain  = sol.domain
                pos_seq = sol.positionSequence
                vel_seq = sol.velocitySequence
                plot!(
                    dist_plot,
                    domain ./ PERIOD,
                    pos_seq .|> only,
                    label  = false #= latexstring("\\mathcal{C} = 2^{$coarse}, \\mathcal{F} = 2^{$fine}") =#,
                    leg    = :bottomleft,
                    xlabel = L"t / T",
                    ylabel = L"\theta",
                    annotation = [(-15, 0, "(b)")]
                )
            end
        end
    end

    plot(
        single_plot,
        gpu_plot,
        dist_plot;
        layout = (3,1),
        link = :x
    )
end

function main()
    coarse_vector = 14:14 |> collect
    fine_vector   = 14:14 |> collect
    eff_matrix = calculate_efficiency(coarse_vector, fine_vector)
    plot_eff(coarse_vector, fine_vector, eff_matrix)
end

# const DATADIR = dirname(@__DIR__) * "/data/"
# const GRAVITY              = 9.81
# const RODLENGTH            = GRAVITY
# const PERIOD               = sqrt(RODLENGTH / GRAVITY)
# const INITIALPOSITION      = Float32[0.]
# const INITIALVELOCITY      = Float32[1.]

# the  true energy is equal to initial energy, because it doesn't change
# in this case the initial energy is just the kinetic energy because the bob is at the bottom
# energy in terms of mass, will get canceled when comparing to simulated
# and I don't want to write the potential energy
# const init_pot_energy = GRAVITY * RODLENGTH * (1 - cos(sum(abs2, INITIALPOSITION)))
# const init_kin_energy = 0.5 * sum(abs2, INITIALVELOCITY)
# const TRUE_ENERGY     = init_kin_energy + init_pot_energy

main()
# coarse_vector = 3:14 |> collect
# fine_vector   = 3:14 |> collect
# single_bench  = load(DATADIR * "bench_single.jld2")
# gpu_bench     = load(DATADIR * "bench_gpu.jld2"; nested = true)
# dist_bench    = load(DATADIR * "bench_dist.jld2"; nested = true)
# eff_matrix    = calculate_efficiency(coarse_vector, fine_vector, dist_bench)
# zeroed_em = map(x -> isapprox(x, 0.0; atol = 10^-10) ? NaN : x, eff_matrix)
# abs_zem = abs.(zeroed_em)
