module Parareal

export euler, verlet                 # integration.jl
export Interval, FirstOrderIVP, SecondOrderIVP # ivp.jl
export Propagator
export parareal                      # Parareal.jl

include("integration.jl")
include("ivp.jl")
include("convergence.jl")
include("discretization.jl")
include("propagation.jl")

"""
    parareal(ivp :: T, coarsePropagator :: Propagator, finePropagator :: Propagator) where T <: InitialValueProblem

Numerically solve the given initial value problem in parallel using a given
propagator and discretizations.
"""
function parareal(ivp :: T, coarsePropagator :: Propagator, finePropagator :: Propagator) where T <: InitialValueProblem
    # Some notes on terminology and consistency
    # IVP.................A structure representing an initial value problem
    #                     consisting of a derivative function, an initial value
    #                     and a domain (see Domain below)
    # Domain..............the interval on which an IVP is defined
    # Subdomain...........a domain that is a subset of another domain
    # Discretized Domain..a vector of points, all of which are in a domain
    # Range...............
    # Discretized Range...a vector of points corresponding to the output of the
    #                     solution function
    # Solution............an ordered pair of the discretized domain and the
    #                     discretized range that satisfies the original IVP
    # Propagator..........a structure consisting of a numerical integrator
    #                     and the number of points on which to evaluate
    coarseDiscretization = coarsePropagator.discretization

    # create a bunch of sub-intervals on which to parallelize
    subDomains = partition(ivp.domain, coarseDiscretization)
    # INITIAL PROPAGATION
    # effectively creating an initial value for each sub-interval
    # same as the end of the loop but with all correctors equal to zero
    initialSolution = propagate(ivp, coarsePropagator)
    discretizedDomain, discretizedRange = initialSolution.domain, initialSolution.range
    # create a bunch of smaller initial value problems that can be solved in parallel
    subProblems = T.(ivp.der, discretizedRange[1:end-1], subDomains) # FIXME: generalize to include SecondOrderIVP

    # allocate space
    subSolutionCoarse = similar(subDomains, Solution)
    subSolutionFine   = similar(subDomains, Solution)
    correctors        = similar(discretizedRange)
    # LOOP PHASE
    for iteration in 1:coarseDiscretization # while # TODO: add convergence criterion
        # println("Iteration $iteration")

        # PARALLEL COARSE
        Threads.@threads for i in eachindex(subProblems)
            # println("Coarse subdomain $i is running on thread ", Threads.threadid())
            subSolutionCoarse[i] = propagate(subProblems[i], coarsePropagator)
        end

        # PARALLEL FINE
        Threads.@threads for i in eachindex(subProblems)
            # println("Fine subdomain $i is running on thread ", Threads.threadid())
            subSolutionFine[i] = propagate(subProblems[i], finePropagator)
        end

        # CORRECTORS
        for i in eachindex(subProblems)
            correctors[i] = subSolutionFine[i].range[end] - subSolutionCoarse[i].range[end]
        end
        # CORRECTION PHASE
        discretizedRange = propagate(ivp, coarsePropagator, correctors).range
        subProblems         = T.(ivp.der, discretizedRange[1:end-1], subDomains)
    end
    return [discretizedDomain, discretizedRange]
end
end
