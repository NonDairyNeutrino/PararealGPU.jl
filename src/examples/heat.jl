# This is an example of using the Parareal algorithm to solve the simple 
# initial value problem of du/dt = u with u(0) = u0
# Author: Nathan Chapman
# Date: 6/29/24
using Plots

# SET UP
include("../structs.jl")
include("../convergence.jl")
include("../discretization.jl")
include("../propagation.jl")

# DEFINE THE INITIAL VALUE PROBLEM
# the derivative function defined in terms of time and the value of the function
function der(t, u)
    return u # this encodes the differential equation du/dt = u
end
const INITIALVALUE = 1.0
const DOMAIN       = Interval(0., 1.)
const ivp = InitialValueProblem(der, INITIALVALUE, DOMAIN)

# define the underlying propagator e.g. euler, RK4, velocity verlet, etc.
"""
    euler(point, slope, step)

The Euler method of numerical integration.
"""
function euler(point, slope, step)
    return point + step * slope
end
propagator = euler

# first we must initialize the algorithm with a coarse solution
# Discretization is the number of sub-domains to use for each time interval
const COARSEDISCRETIZATION = 2^8
const FINEDISCRETIZATION   = COARSEDISCRETIZATION^2
const SUBDOMAINS = partition(ivp.domain, COARSEDISCRETIZATION)

# INITIAL COARSE PROPAGATION
discretizedDomain, solution = coarsePropagate(propagator, DOMAIN, INITIALVALUE)

subDomainCoarse     = similar(SUBDOMAINS, Vector{Vector{Float64}})
subDomainFine       = similar(SUBDOMAINS, Vector{Vector{Float64}})
subDomainCorrectors = similar(solution)
# LOOP PHASE
for iteration in 1:COARSEDISCRETIZATION # while # TODO: add convergence criterion
    println("Iteration $iteration")
    # PARALLEL COARSE
    Threads.@threads for i in eachindex(SUBDOMAINS)
        println("Coarse subdomain $i is running on thread ", Threads.threadid())
        subDomainCoarse[i] = coarsePropagate(propagator, SUBDOMAINS[i], solution[i])
    end

    # PARALLEL FINE
    Threads.@threads for i in eachindex(SUBDOMAINS)
        println("Fine subdomain $i is running on thread ", Threads.threadid())
        subDomainFine[i] = finePropagate(propagator, SUBDOMAINS[i], solution[i])
    end

    # CORRECTORS
    for subdomain in eachindex(SUBDOMAINS)
        subDomainCorrectors[subdomain] = subDomainFine[subdomain][2][end] - subDomainCoarse[subdomain][2][end]
    end
    # CORRECTION PHASE
    global solution = correct(propagator, subDomainCorrectors)
end

include("../plotting.jl")
