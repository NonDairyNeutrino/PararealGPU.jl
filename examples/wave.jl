# This is an example of using the Parareal algorithm to solve the simple 
# initial value problem of d^2 u / dt^2 = -u with u(0) = u0, u'(0) = v0
# Author: Nathan Chapman
# Date: 7/10/24

include("../src/Parareal.jl")
using Plots, .Parareal

# DEFINE THE SECOND-ORDER INITIAL VALUE PROBLEM
"""
    acceleration(position :: Float64, velocity :: Float64) :: Float64

Define the acceleration in terms of the given differential equation.
"""
function acceleration(position :: Float64, velocity :: Float64) :: Float64
    return -position # this encodes the differential equation u''(t) = -u
end

const INITIALPOSITION = 1.
const INITIALVELOCITY = 0.
const DOMAIN          = Interval(0., 2 * pi)
const IVP             = SecondOrderIVP(acceleration, INITIALPOSITION, INITIALVELOCITY, DOMAIN) # second order initial value problem

# DEFINE THE COARSE AND FINE PROPAGATION SCHEMES
const PROPAGATOR           = velocityVerlet
const COARSEDISCRETIZATION = Threads.nthreads()     # 1 region per core
const FINEDISCRETIZATION   = 2 * Threads.nthreads()

const COARSEPROPAGATOR = Propagator(PROPAGATOR, COARSEDISCRETIZATION)
const FINEPROPAGATOR   = Propagator(PROPAGATOR, FINEDISCRETIZATION)

discretizedDomain, discretizedRange = parareal(IVP, COARSEPROPAGATOR, FINEPROPAGATOR)

# plotting
plot(
    discretizedDomain, 
    [discretizedRange, cos.(discretizedDomain)],
    label = ["numeric" "analytic"],
    title = "coarse: $COARSEDISCRETIZATION, fine: $FINEDISCRETIZATION"
)
# Dots to highlight numeric solution
scatter!(discretizedDomain, discretizedRange, label = "")