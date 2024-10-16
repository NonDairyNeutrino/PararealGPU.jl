"""
Parareal.jl provides functionality to solve an initial value problem in parallel through the use of a predictor-corrector scheme.
"""
module PararealGPU

export euler, verlet, velocityVerlet           # integration.jl
export Interval, FirstOrderIVP, SecondOrderIVP # ivp.jl
export Propagator
export parareal                                # Parareal.jl

using CUDA
import Adapt

include("integration.jl")
include("ivp.jl")
include("convergence.jl")
include("discretization.jl")
include("propagation.jl")
include("kernel.jl")

"""
    parareal(ivp :: FirstOrderIVP, coarsePropagator :: Propagator, finePropagator :: Propagator)

Numerically solve the given initial value problem in parallel using a given
propagator and discretizations.
"""
function parareal(ivp :: FirstOrderIVP, coarsePropagator :: Propagator, finePropagator :: Propagator)
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
    subProblems = FirstOrderIVP.(ivp.der, discretizedRange[1:end-1], subDomains)

    # allocate space
    subSolutionCoarse = similar(subDomains, Solution)
    subSolutionFine   = similar(subDomains, Solution)
    correctors        = similar(discretizedRange)
    # LOOP PHASE
    for iteration in 1:coarseDiscretization # while # TODO: add convergence criterion
        # println("Iteration $iteration")

        # PARALLEL COARSE
        kernel = @cuda launch = false pararealKernel(subProblems, coarsePropagator, subSolutionCoarse)
        config = launch_configuration(kernel.fun)
        kernel(subProblems, coarsePropagator, subSolutionCoarse; config.threads, config.blocks)
        # Threads.@threads for i in eachindex(subProblems)
        #     # println("Coarse subdomain $i is running on thread ", Threads.threadid())
        #     subSolutionCoarse[i] = propagate(subProblems[i], coarsePropagator)
        # end

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
        subProblems         = FirstOrderIVP.(ivp.der, discretizedRange[1:end-1], subDomains)
    end
    return [discretizedDomain, discretizedRange]
end

"""
    parareal(ivp :: SecondOrderIVP, coarsePropagator :: Propagator, finePropagator :: Propagator)

Numerically solve the given initial value problem in parallel using a given
propagator and discretizations.
"""
function parareal(ivp :: SecondOrderIVP, coarsePropagator :: Propagator, finePropagator :: Propagator)
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
    
# TODO: create structs with fields first passed to the device
# e.g. SecondOrderIVP(ivp.acceleration, cu(discretizedRange[1]), ...)
    coarseDiscretization = coarsePropagator.discretization

    # create a bunch of sub-intervals on which to parallelize
    subDomains = partition(ivp.domain, coarseDiscretization)
    # INITIAL PROPAGATION
    # effectively creating an initial value for each sub-interval
    # same as the end of the loop but with all correctors equal to zero
    initialSolution = propagate(ivp, coarsePropagator)
    discretizedDomain, discretizedRange, discretizedVelocity = initialSolution.domain, initialSolution.range, initialSolution.derivative
    # create a bunch of smaller initial value problems that can be solved in parallel
    subProblems = SecondOrderIVP.(ivp.acceleration, discretizedRange[1:end-1], discretizedVelocity[1:end-1], subDomains)

    # allocate space
    subSolutionCoarse = similar(subDomains, Solution)
    subSolutionFine   = similar(subDomains, Solution)
    rangeCorrectors   = similar(discretizedRange)
    velocityCorrectors= similar(discretizedRange) # length minus 1?
    # LOOP PHASE
    for iteration in 1:coarseDiscretization # while # TODO: add convergence criterion
        # println("Iteration $iteration")

# TODO: change from CPU parallel to GPU parallel with kernel/pararealKernel
        # PARALLEL COARSE

        # Threads.@threads for i in eachindex(subProblems)
        #     # println("Coarse subdomain $i is running on thread ", Threads.threadid())
        #     subSolutionCoarse[i] = propagate(subProblems[i], coarsePropagator)
        # end

        # PARALLEL FINE
        Threads.@threads for i in eachindex(subProblems)
            # println("Fine subdomain $i is running on thread ", Threads.threadid())
            subSolutionFine[i] = propagate(subProblems[i], finePropagator)
        end

        # CORRECTORS
        for i in eachindex(subProblems)
            rangeCorrectors[i]    = subSolutionFine[i].range[end]      - subSolutionCoarse[i].range[end]
        end
        for i in eachindex(subProblems)
            velocityCorrectors[i] = subSolutionFine[i].derivative[end] - subSolutionCoarse[i].derivative[end]
        end
        # CORRECTION PHASE
        corrected   = propagate(ivp, coarsePropagator, rangeCorrectors, velocityCorrectors)
        discretizedRange, discretizedVelocity = corrected.range, corrected.derivative
        subProblems = SecondOrderIVP.(ivp.acceleration, discretizedRange[1:end-1], discretizedVelocity[1:end-1], subDomains)
    end
    return [discretizedDomain, discretizedRange]
end
end
