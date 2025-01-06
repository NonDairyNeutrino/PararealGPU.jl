using Test

include("../src/subproblems.jl")
include("../src/integration.jl")

# SETUP
"""
    acceleration(position :: Float64, velocity :: Float64) :: Float64

Define the acceleration in terms of the given differential equation.
"""
function acceleration(position :: Vector{Float64}, velocity :: Vector{Float64}) :: Vector{Float64}
    return -position # this encodes the differential equation u''(t) = -u
end

const INITIALPOSITION = [0.]
const INITIALVELOCITY = [1.]
const DOMAIN         = Interval(0., 2 * pi)
const IVP      = SecondOrderIVP(DOMAIN, acceleration, INITIALPOSITION, INITIALVELOCITY) # second order initial value problem

# DEFINE THE COARSE AND FINE PROPAGATION SCHEMES
const PROPAGATOR           = sympecticEuler
const COARSEDISCRETIZATION = 2^0 * Threads.nthreads() # 1 region per core
const COARSEPROPAGATOR = Propagator(PROPAGATOR, COARSEDISCRETIZATION)

# TESTING
# println("Testing partition")
# subDomainVector = partition(IVP.domain, COARSEPROPAGATOR.discretization)
# display(subDomainVector)

println("Testing propagate")
initialSolution = propagate(IVP, COARSEPROPAGATOR)
display(initialSolution)
