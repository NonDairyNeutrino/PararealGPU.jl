module Parareal

export parareal

include("structs.jl")
include("convergence.jl")
include("discretization.jl")
include("propagation.jl")

function parareal(
    ivp :: InitialValueProblem,
    propagator :: Fucntion,
    coarseDiscretization :: Int,
    fineDiscetization :: Int)

    # create a bunch of sub-intervals on which to parallelize
    subDomains = partition(ivp.domain, coarseDiscretization)
    # INITIAL PROPAGATION
    # effectively creating an initial value for each sub-interval
    # same as the end of the loop but with all correctors equal to zero
    coarsePropagator = Propagator(propagator, coarseDiscretization)
    discretizedDomain, solution = propagate(ivp, coarsePropagator)
    # create a bunch of smaller initial value problems that can be solved in parallel
    subProblems = InitialValueProblem.(ivp.der, solution, subDomains)

    subDomainCoarse     = similar(subDomains, Vector{Vector{Float64}})
    subDomainFine       = similar(subDomains, Vector{Vector{Float64}})
    subDomainCorrectors = similar(solution)
    # LOOP PHASE
    for iteration in 1:coarseDiscretization # while # TODO: add convergence criterion
        println("Iteration $iteration")
        # TODO: change propagation to use sub-IVPs
        # PARALLEL COARSE
        Threads.@threads for i in eachindex(subDomains)
            println("Coarse subdomain $i is running on thread ", Threads.threadid())
            subDomainCoarse[i] = propagate(propagator, subDomains[i], solution[i])
        end

        # PARALLEL FINE
        Threads.@threads for i in eachindex(subDomains)
            println("Fine subdomain $i is running on thread ", Threads.threadid())
            subDomainFine[i] = finePropagate(propagator, subDomains[i], solution[i])
        end

        # CORRECTORS
        for subdomain in eachindex(subDomains)
            subDomainCorrectors[subdomain] = subDomainFine[subdomain][2][end] - subDomainCoarse[subdomain][2][end]
        end
        # CORRECTION PHASE
        global solution = correct(propagator, subDomainCorrectors)
    end
end
