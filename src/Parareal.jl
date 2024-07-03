module Parareal

export Interval, InitialValueProblem, parareal

include("structs.jl")
include("convergence.jl")
include("discretization.jl")
include("propagation.jl")

function parareal(
    ivp :: InitialValueProblem,
    propagator :: Function,
    coarseDiscretization :: Int,
    fineDiscetization :: Int)

    # create a bunch of sub-intervals on which to parallelize
    subDomains = partition(ivp.domain, coarseDiscretization)
    # INITIAL PROPAGATION
    # effectively creating an initial value for each sub-interval
    # same as the end of the loop but with all correctors equal to zero
    coarsePropagator = Propagator(propagator, coarseDiscretization)
    finePropagator   = Propagator(propagator, fineDiscetization)
    discretizedDomain, solution = propagate(ivp, coarsePropagator)
    println(length(solution))
    # create a bunch of smaller initial value problems that can be solved in parallel
    subProblems = InitialValueProblem.(ivp.der, solution, subDomains)

    # allocate space
    subDomainCoarse     = similar(subDomains, Vector{Vector{Float64}})
    subDomainFine       = similar(subDomains, Vector{Vector{Float64}})
    subDomainCorrectors = similar(solution)
    # LOOP PHASE
    for iteration in 1:coarseDiscretization # while # TODO: add convergence criterion
        # println("Iteration $iteration")

        # PARALLEL COARSE
        Threads.@threads for i in eachindex(subProblems)
            # println("Coarse subdomain $i is running on thread ", Threads.threadid())
            subDomainCoarse[i] = propagate(subProblems[i], coarsePropagator)
        end

        # PARALLEL FINE
        Threads.@threads for i in eachindex(subProblems)
            # println("Fine subdomain $i is running on thread ", Threads.threadid())
            subDomainFine[i] = finePropagate(subProblems[i], finePropagator)
        end

        # CORRECTORS
        for i in eachindex(subProblems)
            subDomainCorrectors[i] = subDomainFine[i][2][end] - subDomainCoarse[i][2][end]
        end
        # CORRECTION PHASE
        solution = propagate(ivp, coarsePropagator, subDomainCorrectors)
    end
    return [discretizedDomain, solution]
end
end
