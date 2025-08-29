# The total energy of simple harmonic oscillator, here a pendulum, does not change according to
# conservation of energy.
# Due to error arising from sources such as float-point approximation, compound error from the
# discretization of derivatives, etc.
# Here we investigate the dependence of the energy drift on the number of threads.

using Plots: plot, plot!, savefig
using LaTeXStrings, JLD2
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU
include("analysis.jl")

# DEFINE MODEL PARAMETERS
# the frequency (spatial or temporal) constrains the potential values for the length of the rod
# because we're leaving the frequency at 1, and assuming we're on Earth, the length of the rod must
# be 9.8m long.
const MASS                 = 1
const GRAVITY              = 1 # 9.81
const RODLENGTH            = GRAVITY
const INITIALPOSITION      = Float32[0.]
const INITIALVELOCITY      = Float32[1.]
# the  true energy is equal to initial energy, because it doesn't change
# in this case the initial energy is just the kinetic energy because the bob is at the bottom
# energy in terms of mass, will get canceled when comparing to simulated
# and I don't want to write the potential energy
"""
    0.5 * MASS * sum(abs2, INITIALVELOCITY)

The initial energy of the simulation
"""
const INITIALENERGY = 0.5 * MASS * sum(abs2, INITIALVELOCITY) # = ||v||^2

"""
    plot_energy_time(bench_file :: String, coarse :: Int, fine :: Int; drawpotential = false, drawkinetic = false)

Calculate and plot the energy over the simulated time.
"""
function plot_energy_time(bench_file :: String, coarse :: Int, fine :: Int; drawpotential = false, drawkinetic = false)
    bench      = load_bench(bench_file, coarse, fine)
    sol        = bench.value[1]
    pos        = sol.positionSequence
    vel        = sol.velocitySequence
    ke         = calculate_kinetic_energy.(vel; mass = MASS)
    pe         = calculate_potential_energy.(pos; gravity = GRAVITY, rod_length = RODLENGTH)
    me         = ke + pe

    initial    = fill(INITIALENERGY, length(sol.domain))
    data       = [initial me]
    labels     = ["Initial" "Total"]
    style      = drawkinetic || drawpotential ? [:dash :dash] : [:solid :solid]
    if drawkinetic
        data   = [data ke];
        labels = [labels "Kinetic"]
        style  = [style :solid]
    end
    if drawpotential
        data   = [data pe];
        labels = [labels "Potential"]
        style  = [style :solid]
    end

    plt        = plot(
        sol.domain ./ 2pi,
        data / INITIALENERGY;
        xticks = 0:10,
        labels = labels,
        xlabel = L"t/T", 
        ylabel = L"E/E_0",
        style  = style,
        linewidth = 2,
        size = (h -> (MathConstants.golden * h, h))(400) # (733, 567) # <-- aspect ratio of Letter paper 
    )
    return plt
end

"""
    get_errors(bench_dict :: Dict; initial_energy = INITIALENERGY) :: Tuple{Vector{Int}, Vector{Float64}}

Get the discretization and error for a given set of benchmarks.
"""
function get_errors(bench_dict :: Dict; initial_energy = INITIALENERGY) :: Tuple{Vector{Int}, Vector{Float64}}
    disc_vec  = (k -> 2^parse(Int, k)).(keys(bench_dict))
    sol_vec   = (b -> b.value[1]).(values(bench_dict))
    err_vec   = calculate_error(initial_energy).(sol_vec)

    ordering  = sortperm(disc_vec)
    disc_vec .= disc_vec[ordering]
    err_vec  .= err_vec[ordering]

    return disc_vec, err_vec
end

"""
    plot_error_disc(bench_dict :: Dict; initial_energy = INITIALENERGY)

TBW
"""
function plot_error_disc(bench_dict :: Dict; initial_energy = INITIALENERGY)
    disc_vec, err_vec = get_errors(bench_dict; initial_energy = initial_energy)
    plt = plot(
        disc_vec,
        err_vec,
        xticks = disc_vec,
        xscale = :log2,
        xlabel = L"N_\mathcal{F}",
        ylabel = L"(E - E_0) / E_0",
        labels = false
    )
    return plt
end

"""
    plot_error_disc(bench_file :: String, coarse :: Int, fine :: Colon; initial_energy = INITIALENERGY)

TBW
"""
function plot_error_disc(bench_file :: String, coarse :: Int, fine :: Colon; initial_energy = INITIALENERGY)
    fine_bench_dict = load(bench_file; nested = true)["$coarse"]
    plt = plot_error_disc(fine_bench_dict; initial_energy = initial_energy)
    return plt
end

function plot_error_disc(bench_file :: String, coarse :: Colon, fine :: Int)
    
end

"""
    plot_error_disc(bench_file :: String, coarse_vec :: Vector{Int}, fine :: Colon; initial_energy = INITIALENERGY)

TBW
"""
function plot_error_disc(bench_file :: String, coarse_vec :: Vector{Int}, fine :: Colon; initial_energy = INITIALENERGY)
    all_benches         = load(bench_file; nested = true)
    fine_bench_dict_vec = getindex.(Ref(all_benches), string.(coarse_vec))
    disc_err_tup_vec    = get_errors.(fine_bench_dict_vec; initial_energy = initial_energy)
    disc_mat            = first.(disc_err_tup_vec) |> stack
    err_mat             = Base.Fix2(getindex, 2).(disc_err_tup_vec) |> stack
    plt                 = plot(
        disc_mat, 
        err_mat,
        xticks = disc_mat[:, 1],
        xscale = :log2,
        xlabel = L"N_\mathcal{F}",
        ylabel = L"(E - E_0) / E_0",
        labels = (c -> latexstring("N_\\mathcal{C} = $c")).(permutedims(coarse_vec))
    )
    return plt
end

function plot_error_finedisc(bench_file :: String; initial_energy = INITIALENERGY)
    coarse_bench_dict   = load(bench_file; nested = true)
    coarse_disc_vec     = keys(coarse_bench_dict)   |> collect .|> Base.Fix1(parse, Int)
    fine_bench_dict_vec = values(coarse_bench_dict) |> collect
    ordering            = sortperm(coarse_disc_vec)
    coarse_disc_vec     = coarse_disc_vec[ordering]
    fine_bench_dict_vec = fine_bench_dict_vec[ordering]

    disc_err_tup_vec    = get_errors.(fine_bench_dict_vec; initial_energy = initial_energy)
    disc_mat            = first.(disc_err_tup_vec) |> stack
    err_mat             = Base.Fix2(getindex, 2).(disc_err_tup_vec) |> stack
    plt                 = plot(
        disc_mat, 
        err_mat,
        xticks = disc_mat[:, 1],
        xscale = :log2,
        xlabel = L"N_\mathcal{F}",
        ylabel = L"(E - E_0) / E_0",
        labels = (c -> latexstring("N_\\mathcal{C} = $c")).(permutedims(coarse_disc_vec))
    )
    return plt
end

plot_error_disc(bench_file :: String, coarse :: Colon, fine :: Colon) = plot_error_finedisc(bench_file)