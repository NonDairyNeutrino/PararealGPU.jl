# This is an example of using the Parareal algorithm to solve the simple 
# initial value problem of d^2 u / dt^2 = -u with u(0) = u0, u'(0) = v0
# Author: Nathan Chapman
# Date: 7/10/24

include("../src/Parareal.jl")
using Plots, .Parareal

# DEFINE THE SECOND-ORDER INITIAL VALUE PROBLEM
function accerlation(position :: Vector{Float64}, velocity :: Vector{Float64}) :: Vector{Float64}
    return -position # this encodes the differential equation u''(t) = -u
end

const INITIALPOSITION = 0
const INITIALVELOCITY = 0
const IVP             = # second order initial value problem

propagtor = verlet

const COARSEDISCRETIZATION = Threads.nthreads()
const FINEDISCRETIZATION   = 2 * Threads.nthreads()

discretizedDomain, discretizedRange = parareal(IVP, propagator, COARSEDISCRETIZATION, FINEDISCRETIZATION)

# plotting
include("../src/plotting.jl")