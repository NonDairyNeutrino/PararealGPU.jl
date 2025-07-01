#= 
Compare the timings of the implementation between how many threads are used.
Comparing between
- CPU single threaded
    - just straight velocityVerlet
- CPU multithreaded
    - CPU parareal
- GPU multithreaded
    - local GPU parareal
- distributed
    - distributed parareal
=#

using Plots: plot, plot!, savefig
using BenchmarkTools
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU

# DEFINE CLUSTER
const NODEVECTOR           = String["Electromagnetism"]
# DEFINE COMPUTATIONAL PARAMETERS
const COARSEINTEGRATOR     = symplecticEuler
const COARSEDISCRETIZATION = 2^10                        # how many total problems
const FINEINTEGRATOR       = velocityVerlet
const FINEDISCRETIZATION   = 2^10                         # 2^10 = 1024 steps -> each step is ~0.01% of the domain
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

# SINGLE THREADED
const FINE_STEP = (DOMAINUPPERBOUND - DOMAINLOWERBOUND) / FINEDISCRETIZATION

function single(pos0 :: Vector{T}, vel0 :: Vector{T}) :: Tuple{Vector{T}, Vector{T}} where T <: AbstractFloat
    pos = pos0
    vel = vel0
    # cpu single threaded
    for i in 1:FINEDISCRETIZATION
        pos, vel = FINEINTEGRATOR(pos, vel, (x, v) -> -WAVENUMBER * x, FINE_STEP)
    end
    return pos, vel
end

# single()
display(@benchmark single(INITIALPOSITION, INITIALVELOCITY))
