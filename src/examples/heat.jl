# This is an example of using the Parareal algorithm to solve the simple 
# initial value problem of du/dt = u with u(0) = u0
# Author: Nathan Chapman
# Date: 6/29/24

include("../Parareal.jl")
using Plots, .Parareal

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
const COARSEDISCRETIZATION = Threads.nthreads()
const FINEDISCRETIZATION   = COARSEDISCRETIZATION * 2^2

discretizedDomain, solution = parareal(ivp, propagator, COARSEDISCRETIZATION, FINEDISCRETIZATION)
display(discretizedDomain)
display(solution)
include("../plotting.jl")
