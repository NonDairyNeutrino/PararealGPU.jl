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

const DATADIR = "../benchmarks/data/"

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
const INITIALENERGY = (1//2) * MASS * sum(abs2, INITIALVELOCITY) # = ||v||^2

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

function plot_error_disc(bench_file :: String, coarse :: Int, fine :: Int)
    bench_dist = load(bench_file; nested = true)
    for coarse in keys(bench_dist)
        bench_coarse = bench_dist[coarse]
        for fine in keys(bench_coarse)
            bench = bench_coarse[fine]
            # TODO: calculate error here
        end
    end
    sol   = bench.value[1]
    calculate_error(INITIALENERGY, sol)
end