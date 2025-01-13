# This is an example of using the Parareal algorithm to solve the simple
# initial value problem of d^2 u / dt^2 = -u with u(0) = u0, u'(0) = v0
# Author: Nathan Chapman
# Date: 01/07/25

include("../src/PararealGPU.jl")
using .PararealGPU
using Plots
# SETUP
"""
    acceleration(position :: Float64, velocity :: Float64) :: Float64

Define the acceleration in terms of the given differential equation.
"""
@inline function acceleration(position :: T, velocity :: T) :: T where T
    return -position # this encodes the differential equation u''(t) = -u
end

const INITIALPOSITION = [0.]
const INITIALVELOCITY = [1.]
const DOMAIN         = Interval(0., 2^2 * pi)
const IVP      = SecondOrderIVP(DOMAIN, acceleration, INITIALPOSITION, INITIALVELOCITY) # second order initial value problem

# DEFINE THE COARSE AND FINE PROPAGATION SCHEMES
const INITIALDISCRETIZATION        = Threads.nthreads()
const COARSEPROPAGATOR = Propagator(symplecticEuler, INITIALDISCRETIZATION)
const FINEPROPAGATOR   = Propagator(velocityVerlet,  2^1 * INITIALDISCRETIZATION)

rootSolution = parareal(IVP, COARSEPROPAGATOR, FINEPROPAGATOR)

plot(
    rootSolution.domain,
    [rootSolution.positionSequence .|> first, rootSolution.velocitySequence .|> first],
    label = ["position" "velocity"],
    title = "propagator: $(FINEPROPAGATOR.propagator), discretization: $INITIALDISCRETIZATION"
)
savefig("cos.png")
