# collection of tools to analyze the stability of this implementation

include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU

# DEFINE CLUSTER
const NODEVECTOR           = String["Electromagnetism"]
# DEFINE COMPUTATIONAL PARAMETERS
const COARSEINTEGRATOR     = symplecticEuler
const COARSEDISCRETIZATION = 2^13                        # how many total problems
const FINEINTEGRATOR       = velocityVerlet
const FINEDISCRETIZATION   = 2^8                         # 2^7 = 128 steps -> each step is ~1% of the domain
# DEFINE MODEL PARAMETERS
const WAVENUMBER           = 1.0f0 # * pi
# ACCELERATION(r, v)         = -WAVENUMBER^2 * r         # simple harmonic oscillator
const DOMAINLOWERBOUND, DOMAINUPPERBOUND = 0.0f0, 200.0f0 * pi
const INITIALPOSITION      = Float32[0.]
const INITIALVELOCITY      = Float32[1.]

const TRUEDOMAIN                 = range(DOMAINLOWERBOUND, DOMAINUPPERBOUND, COARSEDISCRETIZATION + 1)
const TRUEPOSITION, TRUEVELOCITY = sin.(WAVENUMBER .* TRUEDOMAIN), cos.(WAVENUMBER .* TRUEDOMAIN)

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

using Plots: plot, plot!, savefig
plot(
    solution.domain,
    [only.(solution.positionSequence) ./ TRUEPOSITION .- 1, only.(solution.velocitySequence) ./ TRUEVELOCITY .- 1],
    title = "%error - coarse: $COARSEDISCRETIZATION, fine: $FINEDISCRETIZATION",
    label = ["position" "velocity"]
)
plot_dir  = string(pwd(), "/")
plot_name = "stability_cd$(COARSEDISCRETIZATION)_fd$(FINEDISCRETIZATION)_ub$(DOMAINUPPERBOUND/pi)pi.pdf"
plot_abs_path = plot_dir * plot_name
println("Plot saved at ", plot_abs_path)
savefig(plot_abs_path)
run(`codium $plot_abs_path`)
