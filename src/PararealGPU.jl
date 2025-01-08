"""
Parareal.jl provides functionality to solve an initial value problem in parallel through the use of a predictor-corrector scheme.
"""
module PararealGPU

export euler, symplecticEuler, velocityVerlet  # integration.jl
export Interval, FirstOrderIVP, SecondOrderIVP # ivp.jl
export Propagator
export parareal                                # Parareal.jl

using CUDA
using LinearAlgebra: norm

include("ivp.jl")
include("discretization.jl")
include("propagation.jl")
include("subproblems.jl")
include("integration.jl")
include("correction.jl")
include("convergence.jl")

"""
    parareal(ivp :: SecondOrderIVP, coarsePropagator :: Propagator, finePropagator :: Propagator)

Numerically solve the given initial value problem in parallel using a given
propagator and discretizations.
"""
function parareal(ivp :: SecondOrderIVP, coarsePropagator :: Propagator, finePropagator :: Propagator; threshold = 10^(-10))
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
    rootSolution, subProblemVector = initializeSubproblems(ivp, coarsePropagator)

    initialDiscretization        = coarsePropagator.discretization
    subSolutionCoarseVector = similar(subProblemVector, Solution)
    subSolutionFineVector   = similar(subProblemVector, Solution)
    positionCorrectorVector = similar(subProblemVector, Vector{Float64})
    velocityCorrectorVector = similar(subProblemVector, Vector{Float64})

    # for iteration in 1:initialDiscretization # parareal converges in at most INITIALDISCRETIZATION iterations
    iteration = 0
    maxIterations    = coarsePropagator.discretization
    oldSolution      = rootSolution
    newSolution      = nothing
    while iteration <= maxIterations || !hasConverged(oldSolution, newSolution; threshold)
        iteration += 1
        oldSolution = newSolution
        print("Beginning iteration $iteration\r")
        # the following loops are disjoint to hopefully take advantage of processor pre-fetching
        # i.e. loop fission

        # coarse propagation
        Threads.@threads for i in eachindex(subProblemVector)
            # this is going to be the CUDA kernel
            subSolutionCoarseVector[i] = propagate(subProblemVector[i], coarsePropagator)
        end

        # fine propagation
        Threads.@threads for i in eachindex(subProblemVector)
            subSolutionFineVector[i] = propagate(subProblemVector[i], finePropagator)
        end

        # correction
        correct!(subSolutionFineVector, subSolutionCoarseVector, positionCorrectorVector, velocityCorrectorVector)

        # correct root solution
        rootSolution = propagate(ivp, coarsePropagator, positionCorrectorVector, velocityCorrectorVector)

        # create new sub problems
        if iteration != initialDiscretization # no need for new subproblems after last iteration
            updateSubproblems!(subProblemVector, rootSolution, ivp.acceleration)
        end
        newSolution = rootSolution
    end
    return rootSolution
end
end
