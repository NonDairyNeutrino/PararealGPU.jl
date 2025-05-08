# collection of tools to analyze the stability of this implementation

using Plots: plot, plot!, savefig
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU

# DEFINE CLUSTER
const NODEVECTOR           = String["Electromagnetism"]
# DEFINE COMPUTATIONAL PARAMETERS
const COARSEINTEGRATOR     = symplecticEuler
const COARSEDISCRETIZATION = 2^13                        # how many total problems
const FINEINTEGRATOR       = velocityVerlet
const FINEDISCRETIZATION   = 2^10                         # 2^10 = 1024 steps -> each step is ~0.01% of the domain
# DEFINE MODEL PARAMETERS
const WAVENUMBER           = 1.0f0 # * pi # DO NO CHANGE
# ACCELERATION(r, v)         = -WAVENUMBER^2 * r         # simple harmonic oscillator
const DOMAINLOWERBOUND     = 0.0f0
# DOMAINUPPERBOUND           = 200.0f0 * pi
const INITIALPOSITION      = Float32[0.]
const INITIALVELOCITY      = Float32[1.]

upperBoundVector = collect(2.0f0 * pi .* (1.0f0:10.0f0))
pos_error        = similar(upperBoundVector, Float32)
vel_error        = similar(upperBoundVector, Float32)
for (i, DOMAINUPPERBOUND) in enumerate(upperBoundVector)
    TRUEDOMAIN                 = range(DOMAINLOWERBOUND, DOMAINUPPERBOUND, COARSEDISCRETIZATION + 1)
    TRUEPOSITION, TRUEVELOCITY = sin.(WAVENUMBER .* TRUEDOMAIN), cos.(WAVENUMBER .* TRUEDOMAIN)

    @info "Beginning " DOMAINUPPERBOUND

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
        threshold = 10.0f0
    )

    # first element is the initial value which always matches the true version
    pos_error[i] = maximum(only.(solution.positionSequence)[2:end] ./ TRUEPOSITION[2:end] .- 1)
    # @assert !isnan(pos_error[i]) "$(count(isnan, percent_errors)) NaNs found; first at $(findfirst(isnan, percent_errors))"
    vel_error[i] = Base.rest(only.(solution.velocitySequence) ./ TRUEVELOCITY .- 1) |> maximum
    @info "Finished with " pos_error[i] vel_error[i]
end

plot(
    upperBoundVector,
    [pos_error, vel_error],
    title = "%error - coarse: $COARSEDISCRETIZATION, fine: $FINEDISCRETIZATION",
    label = ["position" "velocity"]
)
plot_dir  = string(pwd(), "/analysis/images/")
plot_name = "stability_cd$(COARSEDISCRETIZATION)_fd$(FINEDISCRETIZATION).pdf"
plot_abs_path = plot_dir * plot_name
println("Plot saved at ", plot_abs_path)
savefig(plot_abs_path)
# run(`codium $plot_abs_path`)
