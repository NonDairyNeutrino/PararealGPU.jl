# functionality ot prepare the subproblems

"""
    createSubproblems(ivp :: SecondOrderIVP, discretization :: Int) :: Vector{SecondOrderIVP}

Create subproblems from the given initial value problem and discretization.
"""
function initializeSubproblems(
        ivp :: SecondOrderIVP{T}, 
        initialPropagator :: Propagator
    ) :: Tuple{Solution, Vector{SecondOrderIVP{T}}} where T <: AbstractFloat
    # create a bunch of sub-intervals on which to parallelize
    subDomainVector = partition(ivp.domain, initialPropagator.discretization)
    # INITIAL PROPAGATION
    # effectively creating an initial value for each sub-interval
    # same as the end of the loop but with all correctors equal to zero
    initialSolution = propagate(ivp, initialPropagator)

    # create a bunch of smaller initial value problems that can be solved in parallel
    subProblemVector = similar(subDomainVector, SecondOrderIVP{T})
    Threads.@threads for i in eachindex(subDomainVector)
        id                  = ivp.id * "." * string(i)
        subDomain           = subDomainVector[i]
        initialPosition     = initialSolution.positionSequence[i]
        initialVelocity     = initialSolution.velocitySequence[i]
        subProblemVector[i] = SecondOrderIVP(id, subDomain, ivp.acceleration, initialPosition, initialVelocity)
    end
    return initialSolution, subProblemVector
end

"""
    updateSubproblems!(subProblemVector :: Vector{SecondOrderIVP}, rootSolution :: Solution, acceleration :: Function)

Update the current set of subproblems with the corrected root solution.
"""
function updateSubproblems!(
        subProblemVector :: Vector{SecondOrderIVP{T}}, 
        rootSolution :: Solution{T}, 
        acceleration :: Function
    ) where T <: AbstractFloat
    subDomainVector = getproperty.(subProblemVector, :domain) # reuse cause sub-domains don't change
    Threads.@threads for i in eachindex(subDomainVector)
        id                  = subProblemVector[i].id
        subDomain           = subDomainVector[i]               # domain in time
        initialPosition     = rootSolution.positionSequence[i] # initial position at a point in time
        initialVelocity     = rootSolution.velocitySequence[i] # initial velocity at a point in time
        # TODO: make SecondOrderIVP mutable to avoid allocating new problems
        # id, domain, and acceleration don't change
        subProblemVector[i] = SecondOrderIVP{T}(id, subDomain, acceleration, initialPosition, initialVelocity)
    end
    return subProblemVector
end