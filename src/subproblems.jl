# functionality ot prepare the subproblems
include("ivp.jl")
include("discretization.jl")
include("propagation.jl")

"""
    createSubproblems(ivp :: SecondOrderIVP, discretization :: Int) :: Vector{SecondOrderIVP}

Create subproblems from the given initial value problem and discretization.
"""
function initializeSubproblems(ivp :: SecondOrderIVP, initialPropagator :: Propagator) :: Tuple{Solution, Vector{SecondOrderIVP}}
    # create a bunch of sub-intervals on which to parallelize
    subDomainVector = partition(ivp.domain, initialPropagator.discretization)
    # INITIAL PROPAGATION
    # effectively creating an initial value for each sub-interval
    # same as the end of the loop but with all correctors equal to zero
    initialSolution = propagate(ivp, initialPropagator)

    # create a bunch of smaller initial value problems that can be solved in parallel
    subProblemVector = similar(subDomainVector, SecondOrderIVP)
    for i in eachindex(subDomainVector)
        subDomain           = subDomainVector[i]
        initialPosition     = initialSolution.positionSequence[i]
        initialVelocity     = initialSolution.velocitySequence[i]
        subProblemVector[i] = SecondOrderIVP(subDomain, ivp.acceleration, initialPosition, initialVelocity)
    end
    return initialSolution, subProblemVector
end

"""
    updateSubproblems!(subProblemVector :: Vector{SecondOrderIVP}, rootSolution :: Solution, acceleration :: Function)

Update the current set of subproblems with the corrected root solution.
"""
function updateSubproblems!(subProblemVector :: Vector{SecondOrderIVP}, rootSolution :: Solution, acceleration :: Function)
    subDomainVector = getproperty.(subProblemVector, :domain) # reuse cause sub-domains don't change
    for i in eachindex(subDomainVector)
        subDomain           = subDomainVector[i]               # domain in time
        initialPosition     = rootSolution.positionSequence[i] # initial position at a point in time
        initialVelocity     = rootSolution.velocitySequence[i] # initial velocity at a point in time
        subProblemVector[i] = SecondOrderIVP(subDomain, acceleration, initialPosition, initialVelocity)
    end
    return subProblemVector
end