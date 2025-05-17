# functionality ot prepare the subproblems

"""
    createSubproblems(ivp :: SecondOrderIVP, discretization :: Int) :: Vector{SecondOrderIVP}

Create subproblems from the given initial value problem and discretization.
"""
function initializeSubproblems(
        ivp               :: SecondOrderIVP{T}, 
        initialPropagator :: Propagator;
        initialSolution   :: String = ""
    ) :: Tuple{Solution, Vector{SecondOrderIVP{T}}} where T <: AbstractFloat

    if isempty(initialSolution)
        # create a bunch of sub-intervals on which to parallelize
        subDomainVector = partition(ivp.domain, initialPropagator.discretization)
        # INITIAL PROPAGATION
        # effectively creating an initial value for each sub-interval
        # same as the end of the loop but with all correctors equal to zero
        initial_solution = propagate(ivp, initialPropagator)
    else
        # use assert here instead of elseif so an error is thrown
        @assert isfile(initialSolution) "Given checkpoint file does not exist."
        initial_solution = read_checkpoint(initialSolution, T)
        subDomainVector  = let dom = initial_solution.domain
            [Interval(dom[i], dom[i+1]) for i in 1:(length(dom) - 1)]
        end
    end

    # create a bunch of smaller initial value problems that can be solved in parallel
    subProblemVector = similar(subDomainVector, SecondOrderIVP{T})
    Threads.@threads for i in eachindex(subDomainVector)
        id                  = ivp.id * "." * string(i)
        subDomain           = subDomainVector[i]
        initialPosition     = initial_solution.positionSequence[i]
        initialVelocity     = initial_solution.velocitySequence[i]
        subProblemVector[i] = SecondOrderIVP(id, subDomain, ivp.acceleration, initialPosition, initialVelocity)
    end
    return initial_solution, subProblemVector
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