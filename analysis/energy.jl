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
using DelimitedFiles
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU

# DEFINE CLUSTER
const NODEVECTOR           = String["Electromagnetism"]
# DEFINE COMPUTATIONAL PARAMETERS
const COARSEINTEGRATOR     = symplecticEuler
# const COARSEDISCRETIZATION = 2^10                        # how many total problems
const FINEINTEGRATOR       = velocityVerlet
# const FINEDISCRETIZATION   = 2^7                         # 2^10 = 1024 steps -> each step is ~0.01% of the domain
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

const finePowerVector   = [6, 8, 10, 12, 14]
const coarsePowerVector = collect(2:14)
const coarseDiscVector  = 2 .^ coarsePowerVector
const fineDiscVector    = 2 .^ finePowerVector
energyErrorMatrix = zeros(length(coarseDiscVector), length(fineDiscVector))

const plot_dir  = string(pwd(), "/analysis/images/")
const plot_ext  = ".png"
const plot_name = replace(string("energy_", "fine_", finePowerVector, "_coarse_", coarsePowerVector[begin], "_", coarsePowerVector[end]), ", " => "_", r"\[|\]" => "")
const plot_abs_path = plot_dir * plot_name * plot_ext

function calculate_error!(energyErrorMatrix :: Matrix, inds :: Dims{2}, solution :: Solution) :: Nothing
    # measure energy drift with respect to the final value
    final_position = solution.positionSequence |> last |> only
    final_velocity = solution.velocitySequence |> last

    kinetic_energy       = 0.5 * sum(abs2, final_velocity)
    potential_energy     = GRAVITY * RODLENGTH * (1 - cos(final_position))
    energy               = kinetic_energy + potential_energy
    energyErrorMatrix[inds...] = energy / TRUE_ENERGY - 1
    return nothing
end

function plot_error(coarseDiscVector, finePowerVector, energyErrorMatrix)
    plot(
            coarseDiscVector,
            energyErrorMatrix .|> abs .|> log10,
            title  = "t_f/2pi = $DOMAINUPPERBOUNDFACTOR",
            # xlabel = "coarse discretization",
            xlabel = "coarse discretization",
            ylabel = "log10(|%err/err[1]|) at t_f",
            xticks = coarseDiscVector,
            xscale = :log2,
            labels = string.("2^", finePowerVector) |> permutedims,
            dpi    = 200,
            size   = (3 * 200, 2 * 200)
    )
end

function main() :: Nothing
    for (f, FINEDISCRETIZATION) in enumerate(fineDiscVector)
        for (c, COARSEDISCRETIZATION) in enumerate(coarseDiscVector)
            @info "Beginning coarse discretization $COARSEDISCRETIZATION"
            @info "Beginning fine   discretization $FINEDISCRETIZATION"

            solution, _ = solve(
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
            calculate_error!(energyErrorMatrix, (c,f), solution)
            open("analysis/error_data.tsv", "w") do io
                writedlm(io, energyErrorMatrix)
            end
            alert(string("Finished $f.$c/", length(coarsePowerVector), ".", length(finePowerVector)))

            @views energyErrorMatrix[:, f] ./= energyErrorMatrix[1, f]
            plot_error(coarseDiscVector, finePowerVector, energyErrorMatrix)
            savefig(plot_abs_path)
            println("Plot saved at ", plot_abs_path)
            run(`codium $plot_abs_path`)
        end
    end
    println("Error analysis finished.")
    return nothing
end

main()
