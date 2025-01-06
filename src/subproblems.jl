# functionality ot prepare the subproblems
include("ivp.jl")
include("discretization.jl")
include("propagation.jl")

"""
    createSubproblems(ivp :: SecondOrderIVP, discretization :: Int) :: Vector{SecondOrderIVP}

TBW
"""
function createSubproblems(ivp :: SecondOrderIVP, initialPropagator :: Propagator) :: Vector{SecondOrderIVP}
    # create a bunch of sub-intervals on which to parallelize
    subDomains = partition(ivp.domain, initialPropagator.discretization)
    # INITIAL PROPAGATION
    # effectively creating an initial value for each sub-interval
    # same as the end of the loop but with all correctors equal to zero
    initialSolution = propagate(ivp, initialPropagator)
    discretizedDomain, discretizedRange, discretizedVelocity = initialSolution.domain, initialSolution.position, initialSolution.velocity
    # create a bunch of smaller initial value problems that can be solved in parallel
    subProblems = SecondOrderIVP.(subDomains, ivp.acceleration, discretizedRange[1:end-1], discretizedVelocity[1:end-1])
    return subProblems
end