# The total energy of simple harmonic oscillator, here a pendulum, does not change according to
# conservation of energy.
# Due to error arising from sources such as float-point approximation, compound error from the
# discretization of derivatives, etc.
# Here we investigate the dependence of the energy drift on the number of threads.

using LoggingExtras
logger = TeeLogger(MinLevelLogger(FileLogger("log.log"), Logging.Info), ConsoleLogger())
global_logger(logger)

using Alert
using Plots: plot, plot!, savefig
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU

# DEFINE CLUSTER
const NODEVECTOR           = String["Electromagnetism"]
# DEFINE COMPUTATIONAL PARAMETERS
const COARSEINTEGRATOR     = symplecticEuler
const COARSEDISCRETIZATION = 2^10                        # how many total problems
const FINEINTEGRATOR       = velocityVerlet
# const FINEDISCRETIZATION   = 2^3                         # 2^10 = 1024 steps -> each step is ~0.01% of the domain
# DEFINE MODEL PARAMETERS
# the frequency (spatial or temporal) constrains the potential values for the length of the rod
# because we're leaving the frequency at 1, and assuming we're on Earth, the length of the rod must
# be 9.8m long.
const GRAVITY              = 9.81
const RODLENGTH            = GRAVITY
const WAVENUMBER           = 1.0f0 # * pi # DO NO CHANGE
# ACCELERATION(r, v)         = -WAVENUMBER^2 * r         # simple harmonic oscillator
const DOMAINLOWERBOUND     = 0.0f0
const DOMAINUPPERBOUNDFACTOR = 10
const DOMAINUPPERBOUND     = DOMAINUPPERBOUNDFACTOR * 2.0f0 * pi
const INITIALPOSITION      = Float32[0.]
const INITIALVELOCITY      = Float32[1.]

# the  true energy is equal to initial energy, because it doesn't change
# in this case the initial energy is just the kinetic energy because the bob is at the bottom
# energy in terms of mass, will get canceled when comparing to simulated
# and I don't want to write the potential energy
const TRUE_ENERGY = 0.5 * sum(abs2, INITIALVELOCITY) # = ||v||^2
@info "" TRUE_ENERGY

powerVector       = collect(2:15)
discVector        = 2 .^ powerVector
energyErrorVector = zeros(length(powerVector))
plot_dir  = string(pwd(), "/analysis/images/")
plot_name = "energy_cd$(maximum(powerVector))_$(powerVector[begin])_$(powerVector[end]).png"
plot_abs_path = plot_dir * plot_name
for (i, FINEDISCRETIZATION) in enumerate(discVector)
    @info "Beginning fine discretization $FINEDISCRETIZATION"

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

    # measure energy drift with respect to the final value
    kinetic_energy       = 0.5 * sum(abs2, solution.velocitySequence |> last)
    potential_energy     = GRAVITY * RODLENGTH * (1 - cos(solution.positionSequence |> last |> only))
    energy               = kinetic_energy + potential_energy
    energyErrorVector[i] = energy / TRUE_ENERGY - 1

    alert("Finished $i/$(length(powerVector))")

    plot(
        discVector,
        energyErrorVector ./ energyErrorVector[1],
        title  = "coarse: $COARSEDISCRETIZATION, t_f/2pi = $DOMAINUPPERBOUNDFACTOR",
        xlabel = "fine discretization",
        ylabel = "scaled %energy error at t_f",
        xticks = discVector,
        ylims  = (0, Inf),
        xscale = :log2,
        yscale = :log2,
        legend = false,
        dpi    = 200,
        size   = (3 * 200, 2 * 200)
    )

    savefig(plot_abs_path)
    println("Plot saved at ", plot_abs_path)
    run(`codium $plot_abs_path`)
end