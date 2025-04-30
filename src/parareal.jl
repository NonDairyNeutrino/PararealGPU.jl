# """
#     parareal(ivp :: SecondOrderIVP, coarsePropagator :: Propagator, finePropagator :: Propagator)

# Numerically solve the given initial value problem in parallel using a given
# propagator and discretizations.
# """
# function parareal(ivp :: SecondOrderIVP, coarsePropagator :: Propagator, finePropagator :: Propagator; threshold = 10^(-10))
#     # Some notes on terminology and consistency
#     # IVP.................A structure representing an initial value problem
#     #                     consisting of a derivative function, an initial value
#     #                     and a domain (see Domain below)
#     # Domain..............the interval on which an IVP is defined
#     # Subdomain...........a domain that is a subset of another domain
#     # Discretized Domain..a vector of points, all of which are in a domain
#     # Range...............
#     # Discretized Range...a vector of points corresponding to the output of the
#     #                     solution function
#     # Solution............an ordered pair of the discretized domain and the
#     #                     discretized range that satisfies the original IVP
#     # Propagator..........a structure consisting of a numerical integrator
#     #                     and the number of points on which to evaluate
#     rootSolution, subProblemVector = initializeSubproblems(ivp, coarsePropagator)

#     initialDiscretization   = coarsePropagator.discretization
#     subSolutionCoarseVector = similar(subProblemVector, Solution)
#     subSolutionFineVector   = similar(subProblemVector, Solution)
#     positionCorrectorVector = similar(subProblemVector, Vector{Float64})
#     velocityCorrectorVector = similar(subProblemVector, Vector{Float64})

#     # for iteration in 1:initialDiscretization # parareal converges in at most INITIALDISCRETIZATION iterations
#     iteration        = 0
#     maxIterations    = coarsePropagator.discretization
#     oldSolution      = rootSolution
#     newSolution      = nothing
#     while iteration <= maxIterations || !hasConverged(oldSolution, newSolution; threshold)
#         iteration   += 1
#         oldSolution  = newSolution
#         println("Beginning iteration $iteration")
#         # the following loops are disjoint to hopefully take advantage of processor pre-fetching
#         # i.e. loop fission

#         #FIXME: DON'T COARSE PROPAGATION IN THE KERNEL
#         # STORE AND REUSE THE VALUES CALCULATED IN THE CORRECTION
#         discretizedDomain, positionCoarse, velocityCoarse = kernelPrep(
#             subProblemVector, 
#             coarsePropagator.discretization
#         )
#         println("Beginning coarse parallel propagation")
#         subSolutionCoarseVector = pararealSolution(
#             coarsePropagator.propagator, 
#             ivp.acceleration, 
#             discretizedDomain, 
#             positionCoarse, 
#             velocityCoarse
#         )

#         discretizedDomain, positionFine, velocityFine = kernelPrep(
#             subProblemVector, 
#             finePropagator.discretization # FIXME: should this be finePropagator.discretization ?
#         )
#         println("Beginning fine parallel propagation")
#         subSolutionFineVector = pararealSolution(
#             finePropagator.propagator, 
#             ivp.acceleration, 
#             discretizedDomain, 
#             positionFine, 
#             velocityFine
#         )

#         # correction
#         # STORE AND REUSE THE VALUES CALCULATED IN THE CORRECTION
#         println("Beginning correction")
#         correct!(subSolutionFineVector, subSolutionCoarseVector, positionCorrectorVector, velocityCorrectorVector)

#         # correct root solution
#         rootSolution = propagate(ivp, coarsePropagator, positionCorrectorVector, velocityCorrectorVector)

#         # create new sub problems
#         if iteration != initialDiscretization # no need for new subproblems after last iteration
#             println("Updating subproblems")
#             updateSubproblems!(subProblemVector, rootSolution, ivp.acceleration)
#         end
#         newSolution = rootSolution
#     end
#     return rootSolution
# end

"""
    distribute!(
    devPool          :: CachingPool, 
    coarsePropagator :: Propagator, 
    finePropagator   :: Propagator,
    problemVector    :: Vector{SecondOrderIVP}
    ) :: Vector{Solution}

The parallelization scheme to use from the director process.
"""
function distribute(
    problemVector    :: Vector{SecondOrderIVP},
    coarsePropagator :: Propagator, 
    finePropagator   :: Propagator;
    threshold = 10^(-10)
    ) :: Vector{Solution}
    next_level = myid() == 1 ? MANAGERPOOL : intersect(procs(myid()), DEVPOOL)
    println("Distributing $(length(problemVector)) problems from ", myid(), " to ", next_level)
    solutionVector = pmap(
        ivp -> parareal_recursive(
            # distribute! args
            ivp, 
            coarsePropagator,
            finePropagator;
            threshold = threshold
        ), 
        # if I'm the director, distribute over the managers
        # if I'm a manager, distribute over the device processes on this machine
        CachingPool(next_level),
        problemVector;
        # batch_size = div(length(problemVector), length(next_level)) # load balance
    )
    return solutionVector
end

"""
    gpu!(
    acceleration   :: Function, 
    prop           :: Propagator, 
    problemVector  :: Vector{SecondOrderIVP}, 
    solutionVector :: Vector{Solution}
    ) :: Nothing

The parallelization scheme to use from the worker processes.
"""
function gpu!(
    problemVector  :: Vector{SecondOrderIVP},
    prop           :: Propagator, 
    solutionVector :: Vector{Solution}
    ) :: Nothing
    # prepare arrays to put data into
    discretizedDomain, positionMatrix, velocityMatrix = kernelPrep(
        problemVector, 
        prop.discretization
    )
    # println("Beginning fine parallel propagation")
    acceleration = problemVector[1].acceleration
    solutionVector .= pararealSolution!(
        prop.propagator,
        acceleration, 
        discretizedDomain .|> Float32, 
        positionMatrix .|> Float32, 
        velocityMatrix .|> Float32
    )
    error("STOP")
    return
end
# """
#     getOffloadingScheme(
#     devPool          :: CachingPool, 
#     coarsePropagator :: Propagator, 
#     finePropagator   :: Propagator,
#     problemVector    :: Vector{SecondOrderIVP}, 
#     solutionVector   :: Vector{Solution}
#     ) :: Tuple{Function, Vector{Any}}

# Determine the offloading scheme to use based on whether or not it's run on the director or worker.
# """
# function getOffloadingScheme(
#     devPool          :: CachingPool, 
#     coarsePropagator :: Propagator, 
#     finePropagator   :: Propagator,
#     problemVector    :: Vector{SecondOrderIVP}, 
#     solutionVector   :: Vector{Solution}
#     ) :: Tuple{Function, Vector{Any}}
#     if myid() == 1
#         foo = distribute!
#         args = [devPool, coarsePropagator, finePropagator, problemVector, solutionVector]
#     else
#         foo = gpu!
#         args = [finePropagator, problemVector, solutionVector]
#     end
#     return (foo, args)
# end

function parareal_recursive(
    ivp              :: SecondOrderIVP, 
    coarsePropagator :: Propagator, 
    finePropagator   :: Propagator;
    threshold = 10^(-10)
    )
# ==================================================================================================
    # INITIALIZATION
    # println("Beginning iteration 0 on problem ", ivp.id)
    rootSolution, problemVector = initializeSubproblems(ivp, coarsePropagator)
    predSolution = rootSolution # on iteration 0 predicted solution = root solution

    oldSolution                   = rootSolution
    oldSolution.positionSequence .= zero(oldSolution.positionSequence)
    oldSolution.velocitySequence .= zero(oldSolution.velocitySequence)

    newSolution  = rootSolution

    initialDiscretization   = coarsePropagator.discretization
    subSolutionCoarseVector = similar(problemVector, Solution)
    subSolutionFineVector   = similar(problemVector, Solution)
    positionCorrectorVector = similar(problemVector, Vector{Float64})
    velocityCorrectorVector = similar(problemVector, Vector{Float64})

    # for iteration in 1:initialDiscretization # parareal converges in at most
    # INITIALDISCRETIZATION iterations
    iteration        = 0
    maxIterations    = coarsePropagator.discretization
    
    # choose whether to parallelize over nodes or GPU
    # parallelPropagate, args = getOffloadingScheme(devPool, coarsePropagator, finePropagator, subProblemVector, solutionVector)
# ==================================================================================================
    # BEGIN LOOP
    while iteration < maxIterations || !hasConverged(oldSolution, newSolution; threshold)
        iteration   += 1
        oldSolution  = newSolution

# ==================================================================================================
        # offload and propagate in parallel
        if myid() in MANAGERPOOL || myid() == 1
            println("Beginning iteration ", iteration, " on problem ", ivp.id)
            subSolutionFineVector = distribute(problemVector, coarsePropagator, finePropagator; threshold = threshold)
        else
            gpu!(problemVector, finePropagator, subSolutionFineVector)
        end
# ==================================================================================================
        # correction
        # use the values given by the coarse propagator from the previous iteration
        for (i, (pos, vel)) in enumerate(zip(predSolution.positionSequence[2:end], predSolution.velocitySequence[2:end]))
            # correct! only needs the "last" value in the sequence, so just wrap the value in the an array
            # this emulates the coarse propagator also running for each subproblem because the
            # coarse propagator only takes one step
            # the rest of th
            subSolutionCoarseVector[i] = Solution(Float64[], [pos], [vel])
        end

        # println("Calculating corrections")
        correct!(
            subSolutionFineVector, 
            subSolutionCoarseVector, 
            positionCorrectorVector, 
            velocityCorrectorVector
        )

        # correct root solution
        # println("Coarse propagating")
        rootSolution, predSolution = propagate(
            ivp, 
            coarsePropagator, 
            positionCorrectorVector, 
            velocityCorrectorVector
        )
# ==================================================================================================
        # create new sub problems
        if iteration != initialDiscretization # no need for new subproblems after last iteration
            # println("Updating subproblems")
            updateSubproblems!(
                problemVector, 
                rootSolution, 
                ivp.acceleration
            )
        end
        newSolution = rootSolution
    end
# ==================================================================================================
    return rootSolution
end

function solve(
    ivp :: SecondOrderIVP,
    coarse :: Propagator, 
    fine :: Propagator;
    threshold = 10^(-10)
) :: Solution
    println("Beginning parareal evaluation")
    sol = parareal_recursive(ivp, coarse, fine; threshold = threshold)
    println("Parareal evaluation finished. Closing cluster.")
    # TODO: print stats e.g. total problems, iterations, steps
    rmprocs(workers())
    return sol
end