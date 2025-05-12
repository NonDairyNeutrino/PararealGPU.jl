# collection of tools to analyze the stability of this implementation
using Alert
using Plots: plot, plot!, savefig
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU

# DEFINE CLUSTER
const NODEVECTOR           = String["Electromagnetism"]
# DEFINE COMPUTATIONAL PARAMETERS
const COARSEINTEGRATOR     = symplecticEuler
const COARSEDISCRETIZATION = 2^6                        # how many total problems
const FINEINTEGRATOR       = velocityVerlet
const FINEDISCRETIZATION   = 2^3                         # 2^10 = 1024 steps -> each step is ~0.01% of the domain
# DEFINE MODEL PARAMETERS
# the frequency (spatial or temporal) constrains the potential values for the length of the rod
# because we're leaving the frequency at 1, and assuming we're on Earth, the length of the rod must
# be 9.8m long.
const GRAVITY              = 9.81
const RODLENGTH            = GRAVITY
const WAVENUMBER           = 1.0f0 # * pi # DO NO CHANGE
# ACCELERATION(r, v)         = -WAVENUMBER^2 * r         # simple harmonic oscillator
const DOMAINLOWERBOUND     = 0.0f0
# DOMAINUPPERBOUND           = 200.0f0 * pi
const INITIALPOSITION      = Float32[0.]
const INITIALVELOCITY      = Float32[1.]

# the  true energy is equal to initial energy, because it doesn't change
# in this case the initial energy is just the kinetic energy because the bob is at the bottom
# energy in terms of mass, will get canceled when comparing to simulated
# and I don't want to write the potential energy
const TRUE_ENERGY = 0.5 * sum(abs2, INITIALVELOCITY) # = ||v||^2

upperBoundMultiplierVector = collect(20:-1:1)
upperBoundVector = upperBoundMultiplierVector .* Float32(2pi)
energyErrorVector = zeros(length(upperBoundVector))
const plot_dir  = string(pwd(), "/analysis/images/")
const plot_name = "stability_cd$(COARSEDISCRETIZATION)_fd$(FINEDISCRETIZATION)_tf$(maximum(upperBoundMultiplierVector)).png"
const plot_abs_path = plot_dir * plot_name
for (i, DOMAINUPPERBOUND) in enumerate(upperBoundVector)

    @info "Beginning t_f = $(upperBoundMultiplierVector[i])*2pi"

    solution = solve(
        NODEVECTOR,
        COARSEINTEGRATOR,
        COARSEDISCRETIZATION,
        FINEINTEGRATOR,
        FINEDISCRETIZATION,
        ((r, v) -> -WAVENUMBER^2 * r),
        DOMAINLOWERBOUND,
        DOMAINUPPERBOUND,
        INITIALPOSITION,
        INITIALVELOCITY;
        addlocal  = true,
        # eps(Float32) == 1.1920929f-7
        # sqrt(eps(Float32)) == 0.00034526698f0
        # this mirrors isapprox()
        threshold = sqrt(eps(Float32))
    )
    # first element is the initial value which always matches the true version

    # pos_error[end + 1 - i] = maximum(norm.(solution.positionSequence[2:end] .- TRUEPOSITION[2:end]))
    # # @assert !isnan(pos_error[i]) "$(count(isnan, percent_errors)) NaNs found; first at $(findfirst(isnan, percent_errors))"
    # vel_error[end + 1 - i] = maximum(norm.(solution.velocitySequence[2:end] .- TRUEVELOCITY[2:end]))
    kinetic_energy       = 0.5 * sum(abs2, solution.velocitySequence |> last)
    potential_energy     = GRAVITY * RODLENGTH * (1 - cos(solution.positionSequence |> last |> only))
    energy               = kinetic_energy + potential_energy
    energyErrorVector[end + 1 - i] = energy / TRUE_ENERGY - 1
    # @info "Finished with " pos_error[i] vel_error[i]
    alert("Finished $i/$(length(upperBoundMultiplierVector))")

    plot(
        upperBoundVector |> reverse,
        energyErrorVector #= ./ maximum(energyErrorVector) =#,
        title  = "coarse: $COARSEDISCRETIZATION, fine: $FINEDISCRETIZATION", # log2(max(error)) ~ $(round(Int, log2(maximum(energyErrorVector))))",
        # label  = ["position" "velocity"],
        legend = false,
        xlabel = "t_f/2pi",
        ylabel = "%energy error at t_f",
        xticks = (upperBoundVector, upperBoundMultiplierVector),
        # ylims  = (0, 1),
        # yscale = :log2,
        dpi    = 200,
        size   = (3 * 200, 2 * 200)
    )

    println("Plot saved at ", plot_abs_path)
    savefig(plot_abs_path)
    run(`codium $plot_abs_path`)
end