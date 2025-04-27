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

    initialDiscretization   = coarsePropagator.discretization
    subSolutionCoarseVector = similar(subProblemVector, Solution)
    subSolutionFineVector   = similar(subProblemVector, Solution)
    positionCorrectorVector = similar(subProblemVector, Vector{Float64})
    velocityCorrectorVector = similar(subProblemVector, Vector{Float64})

    # for iteration in 1:initialDiscretization # parareal converges in at most INITIALDISCRETIZATION iterations
    iteration        = 0
    maxIterations    = coarsePropagator.discretization
    oldSolution      = rootSolution
    newSolution      = nothing
    while iteration <= maxIterations || !hasConverged(oldSolution, newSolution; threshold)
        iteration   += 1
        oldSolution  = newSolution
        println("Beginning iteration $iteration")
        # the following loops are disjoint to hopefully take advantage of processor pre-fetching
        # i.e. loop fission

        #FIXME: DON'T COARSE PROPAGATION IN THE KERNEL
        # STORE AND REUSE THE VALUES CALCULATED IN THE CORRECTION
        discretizedDomain, positionCoarse, velocityCoarse = kernelPrep(
            subProblemVector, 
            coarsePropagator.discretization
        )
        println("Beginning coarse parallel propagation")
        subSolutionCoarseVector = pararealSolution(
            coarsePropagator.propagator, 
            ivp.acceleration, 
            discretizedDomain, 
            positionCoarse, 
            velocityCoarse
        )

        discretizedDomain, positionFine, velocityFine = kernelPrep(
            subProblemVector, 
            finePropagator.discretization # FIXME: should this be finePropagator.discretization ?
        )
        println("Beginning fine parallel propagation")
        subSolutionFineVector = pararealSolution(
            finePropagator.propagator, 
            ivp.acceleration, 
            discretizedDomain, 
            positionFine, 
            velocityFine
        )

        # correction
        # STORE AND REUSE THE VALUES CALCULATED IN THE CORRECTION
        println("Beginning correction")
        correct!(subSolutionFineVector, subSolutionCoarseVector, positionCorrectorVector, velocityCorrectorVector)

        # correct root solution
        rootSolution = propagate(ivp, coarsePropagator, positionCorrectorVector, velocityCorrectorVector)

        # create new sub problems
        if iteration != initialDiscretization # no need for new subproblems after last iteration
            println("Updating subproblems")
            updateSubproblems!(subProblemVector, rootSolution, ivp.acceleration)
        end
        newSolution = rootSolution
    end
    return rootSolution
end

function parareal_recursive(
    ivp :: SecondOrderIVP, 
    coarsePropagator :: Propagator, 
    finePropagator :: Propagator, 
    devPool :: CachingPool; 
    threshold = 10^(-10)
    )
# ==================================================================================================
    # INITIALIZATION
    rootSolution, subProblemVector = initializeSubproblems(
        ivp, 
        coarsePropagator
    )
    predSolution = rootSolution # initialize predicted solution

    initialDiscretization   = coarsePropagator.discretization
    subSolutionCoarseVector = similar(subProblemVector, Solution)
    subSolutionFineVector   = similar(subProblemVector, Solution)
    positionCorrectorVector = similar(subProblemVector, Vector{Float64})
    velocityCorrectorVector = similar(subProblemVector, Vector{Float64})

    # for iteration in 1:initialDiscretization # parareal converges in at most
    # INITIALDISCRETIZATION iterations
    iteration        = 0
    maxIterations    = coarsePropagator.discretization
    oldSolution      = rootSolution
    newSolution      = nothing
# ==================================================================================================
    # HOW TO PARALLELIZE

    # if I am the director, distribute the problems over the workers
    # otherwise I am a worker and I solve all the problems on the GPU
    if myid() == 1 # director processes has id == 1
        # parallelize over workers
        function parallelPropagate()
            println("Distributing problems.")
            subSolutionFineVector .= pmap(
                ivp -> parareal_recursive(
                    ivp, 
                    coarsePropagator, 
                    finePropagator,
                    devPool
                ), 
                devPool, 
                subProblemVector
            )
            return subSolutionFineVector
        end
    else
        # parallelize over GPU
        function parallelPropagate()
            println("Executing Order 66")
            discretizedDomain, positionFine, velocityFine = kernelPrep(
                subProblemVector, 
                finePropagator.discretization
            )
            println("Beginning fine parallel propagation")
            subSolutionFineVector .= pararealSolution(
                finePropagator.propagator, 
                ivp.acceleration, 
                discretizedDomain, 
                positionFine, 
                velocityFine
            )
            return subSolutionFineVector
        end
    end
# ==================================================================================================
    # BEGIN LOOP
    while iteration <= maxIterations || !hasConverged(oldSolution, newSolution; threshold)
        iteration   += 1
        oldSolution  = newSolution
        println("Beginning iteration $iteration")

        subSolutionFineVector = parallelPropagate()

        # correction
        # use the values given by the coarse propagator from the previous iteration
        for (i, (pos, vel)) in enumerate(zip(predSolution.positionSequence[2:end], predSolution.velocitySequence[2:end]))
            # correct! only needs the "last" value in the sequence, so just wrap the value in the an array
            # this emulates the coarse propagator also running for each subproblem because the
            # coarse propagator only takes one step
            subSolutionCoarseVector[i] = Solution(Float64[], [pos], [vel])
        end

        println("Beginning correction")
        correct!(
            subSolutionFineVector, 
            subSolutionCoarseVector, 
            positionCorrectorVector, 
            velocityCorrectorVector
        )

        # correct root solution
        rootSolution, predSolution = propagate(
            ivp, 
            coarsePropagator, 
            positionCorrectorVector, 
            velocityCorrectorVector
        )
# ==================================================================================================
        # create new sub problems
        if iteration != initialDiscretization # no need for new subproblems after last iteration
            println("Updating subproblems")
            updateSubproblems!(
                subProblemVector, 
                rootSolution, 
                ivp.acceleration
            )
        end
        newSolution = rootSolution
    end
    return rootSolution
end

function solve(
    coarse :: Propagator, 
    fine :: Propagator, 
    devPool :: CachingPool, 
    ivp :: SecondOrderIVP;
    threshold = 10^(-10)
) :: Solution
    println("Beginning parareal evaluation on workers")
    # TODO: replace parareal with parareal_recursive
    sol = parareal_recursive(ivp, coarse, fine, devPool; threshold = threshold)
    println("Parareal evaluation finished. Closing cluster.")
    rmprocs(workers()...) # TODO: move this out of solve
    return sol
end