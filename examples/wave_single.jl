# This is an example of using the Parareal algorithm to solve the simple
# initial value problem of d^2 u / dt^2 = -u with u(0) = u0, u'(0) = v0
# Author: Nathan Chapman

include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU

# DEFINE CLUSTER
const NODEVECTOR           = String["Electromagnetism"]
# DEFINE COMPUTATIONAL PARAMETERS
const COARSEINTEGRATOR     = symplecticEuler
const COARSEDISCRETIZATION = 2^10 # 2^10 = 1024, 2^15 = 32768, 2^20 = 1048576 # how many total problems
const FINEINTEGRATOR       = velocityVerlet
const FINEDISCRETIZATION   = 2^10  # 2^7 = 128 steps -> each step is ~1% of the domain
# DEFINE MODEL PARAMETERS
const WAVENUMBER           = 1.0f0 # * pi
# ACCELERATION(r, v)         = -WAVENUMBER^2 * r         # simple harmonic oscillator
const DOMAINLOWERBOUND, DOMAINUPPERBOUND = 0.0f0, 2.0f0 * pi
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
    threshold = 1.0f-10
)

include("plot_and_save.jl")
