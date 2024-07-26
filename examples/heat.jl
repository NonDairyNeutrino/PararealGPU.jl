# This is an example of using the Parareal algorithm to solve the simple 
# initial value problem of du/dt = u with u(0) = u0
# Author: Nathan Chapman
# Date: 6/29/24

include("../src/Parareal.jl")
using Plots, .Parareal

# DEFINE THE INITIAL VALUE PROBLEM
# the derivative function defined in terms of time and the value of the function
function der(t, u)
    return u # this encodes the differential equation du/dt = u
end
const INITIALVALUE = 1.0
const DOMAIN       = Interval(0., 1.)
const ivp          = FirstOrderIVP(der, INITIALVALUE, DOMAIN)

# first we must initialize the algorithm with a coarse solution
# Discretization is the number of sub-domains to use for each time interval
const PROPAGATOR           = euler
const COARSEDISCRETIZATION = Threads.nthreads() # 1 region per core
const FINEDISCRETIZATION   = 100

const COARSEPROPAGATOR = Propagator(PROPAGATOR, COARSEDISCRETIZATION)
const FINEPROPAGATOR   = Propagator(PROPAGATOR, FINEDISCRETIZATION)

discretizedDomain, solution = parareal(ivp, COARSEPROPAGATOR, FINEPROPAGATOR)
# display([discretizedDomain, solution])
include("../src/plotting.jl")
