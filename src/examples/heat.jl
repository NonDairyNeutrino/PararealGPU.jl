# This is an example of using the Parareal algorithm to solve the simple 
# initial value problem of du/dt = u with u(0) = u0
# Author: Nathan Chapman
# Date: 6/29/24
using Plots

# FRAMEWORK OF A PROBLEM
"""
    Interval(lb, ub)

An object with lower and upper bounds.
"""
struct Interval
    lb :: Float64
    ub :: Float64
end

"""
    InitialValueProblem(der, initialValue, domain)

An object representing an initial value problem
"""
struct InitialValueProblem
    der :: Function
    initialValue :: Number
    domain :: Interval
end

function discretize(domain :: Interval, discretization :: Int) :: StepRangeLen
    return range(domain.lb, domain.ub, discretization + 1)
end

"""
    partition(domain :: Interval, discretization :: Int)

Partition an interval into a given number of mostly disjoint sub-domains.
"""
function partition(domain :: Interval, discretization :: Int) :: Vector{Interval}
    subdomains = Vector{Interval}(undef, COARSEDISCRETIZATION)
    step       = (domain.ub - domain.lb) / COARSEDISCRETIZATION
    for i in 0:COARSEDISCRETIZATION - 1
        subdomains[i + 1] = Interval(domain.lb + i * step, domain.lb + (i + 1) * step)
    end
    return subdomains
end

# BEGIN DEFINING PROBLEM AT HAND

# begin by defining the initial value problem:
# the derivative function defined in terms of time and the value of the function
function der(t, u)
    return u # this encodes the differential equation du/dt = u
end

const INITIALVALUE = 1.0
const DOMAIN       = Interval(0., 1.)

ivp = InitialValueProblem(der, INITIALVALUE, DOMAIN)

# now that the problem has been defined
# give it to the parareal algorithm

# first we must initialize the algorithm with a coarse solution

# Discretization is the number of sub-domains to use for each time interval
const COARSEDISCRETIZATION = 2^1
const FINEDISCRETIZATION   = (2^2) * COARSEDISCRETIZATION
const SUBDOMAINS = partition(ivp.domain, COARSEDISCRETIZATION)

# here we define the underlying propagator e.g. euler, RK4, velocity verlet, etc.
"""
    euler(point, slope, step)

The Euler method of numerical integration.
"""
function euler(point, slope, step)
    return point + step * slope
end

propagate = euler

"""
    coarsePropagate(domain :: Interval, initialValue :: Float64)

Coarsely propagate a value on an interval.
"""
function coarsePropagate(domain :: Interval, initialValue :: Float64) :: Vector{Vector{Float64}}
    step              = (domain.ub - domain.lb) / COARSEDISCRETIZATION
    discretizedDomain = discretize(domain, COARSEDISCRETIZATION)
    solution          = similar(discretizedDomain)
    solution[1] = initialValue
    for i in 2:COARSEDISCRETIZATION + 1
        solution[i] = propagate(solution[i - 1], der(discretizedDomain[i - 1], solution[i - 1]), step)
    end
    return [discretizedDomain |> collect, solution]
end

"""
    finePropagate(domain :: Interval, initialValue :: Float64)

Finely propagate a value on an interval.
"""
function finePropagate(domain :: Interval, initialValue :: Float64) :: Vector{Vector{Float64}}
    step              = (domain.ub - domain.lb) / FINEDISCRETIZATION
    discretizedDomain = discretize(domain, FINEDISCRETIZATION)
    solution          = similar(discretizedDomain)
    solution[1] = initialValue
    for i in 2:FINEDISCRETIZATION + 1
        solution[i] = propagate(solution[i - 1], der(discretizedDomain[i - 1], solution[i - 1]), step)
    end
    return [discretizedDomain |> collect, solution]
end

discretizedDomain = discretize(DOMAIN, COARSEDISCRETIZATION)
solution = coarsePropagate(DOMAIN, INITIALVALUE)[2]

# now that we have the initial coarse propagation
# parallel iterations can begin
dummyCoarse         = similar(SUBDOMAINS, Vector{Vector{Float64}})
dummyFine           = similar(SUBDOMAINS, Vector{Vector{Float64}})
subDomainCorrectors = similar(solution)
# this loop could be broken into two loops i.e. loop fission
Threads.@threads for i in eachindex(SUBDOMAINS)
    println("Subdomain $i is running on thread ", Threads.threadid())
    coarseDomain, coarseSolution = coarsePropagate(SUBDOMAINS[i], solution[i])
    dummyCoarse[i] = [coarseDomain, coarseSolution]

    fineDomain,   fineSolution   = finePropagate(SUBDOMAINS[i], solution[i])
    dummyFine[i] = [fineDomain, fineSolution]

    subDomainCorrectors[i] = fineSolution[end] - coarseSolution[end]
end

# TODO: add correction phase

plot(
    discretizedDomain, 
    [solution, exp.(discretizedDomain)],
    label = ["numeric" "analytic"]
)                                                                      # Lines
scatter!(discretizedDomain, solution, label = "")                      # Numeric dots
[scatter!(
    dummyCoarse[region][1],
    dummyCoarse[region][2],
    label = "dummyCoarse $region",
    markershape = :rect
    ) for region in eachindex(dummyCoarse)
]
[scatter!(
    dummyFine[region][1],
    dummyFine[region][2],
    label = "dummyFine $region",
    markershape = :diamond
    ) for region in eachindex(dummyFine)
]
scatter!(discretizedDomain, subDomainCorrectors, label = "Correctors") # Correctors
