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
    # println("Distributing $(length(problemVector)) problems from ", myid(), " to ", next_level)
    solutionVector = pmap(
        ivp -> parareal(
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
        batch_size = div(length(problemVector), length(next_level)) # load balance
    )
    return solutionVector
end

"""
    gpu(
        problemVector  :: Vector{SecondOrderIVP},
        prop           :: Propagator
    ) :: Vector{Solution}

The parallelization scheme to use from the worker processes.
"""
function gpu(
    problemVector  :: Vector{SecondOrderIVP},
    prop           :: Propagator
    # solutionVector :: Vector{Solution}
    ) :: Vector{Solution}
    # prepare arrays to put data into
    discretizedDomain, positionMatrix, velocityMatrix = kernelPrep(
        problemVector, 
        prop.discretization
    )
    # println("Beginning fine parallel propagation")
    acceleration = problemVector[1].acceleration
    solutionVector = pararealSolution!(
        prop.propagator,
        acceleration, 
        discretizedDomain .|> Float32, 
        positionMatrix .|> Float32, 
        velocityMatrix .|> Float32
    )
    # error("STOP")
    return solutionVector
end

function batchProblems(problemVector, my_worker_pool)
    batch_size = div(length(problemVector), length(my_worker_pool))
    # if problemVector does not divide evenly into MANAGERPOOL
    # Iterators.partition includes a "remainder" partition; see docs for Iterators.partition
    return Iterators.partition(problemVector, batch_size) .|> collect
end

function parareal(
    ivp              :: SecondOrderIVP, 
    coarsePropagator :: Propagator, 
    finePropagator   :: Propagator;
    threshold = 10^(-10)
    )
# ==================================================================================================
    # INITIALIZATION
    print("Beginning iteration 0\r")
    rootSolution, directorProblemVector = initializeSubproblems(ivp, coarsePropagator)
    predSolution = rootSolution # on iteration 0 predicted solution = root solution
    oldSolution  = rootSolution

    initialDiscretization   = coarsePropagator.discretization
    subSolutionCoarseVector = similar(directorProblemVector, Solution)
    subSolutionFineVector   = similar(directorProblemVector, Solution)
    positionCorrectorVector = similar(directorProblemVector, Vector{Float64})
    velocityCorrectorVector = similar(directorProblemVector, Vector{Float64})

    # for iteration in 1:initialDiscretization # parareal converges in at most
    # INITIALDISCRETIZATION iterations
    iteration        = 0
    maxIterations    = coarsePropagator.discretization
    
    # choose whether to parallelize over nodes or GPU
    # parallelPropagate, args = getOffloadingScheme(devPool, coarsePropagator, finePropagator, subProblemVector, solutionVector)
# ==================================================================================================
    # BEGIN LOOP
    has_converged = false
    while !has_converged && iteration < maxIterations 
        iteration += 1
        print("Beginning iteration $iteration...")
# ==================================================================================================
        # offload and propagate in parallel
            # println("Beginning iteration ", iteration, " on problem ", ivp.id)
            # subSolutionFineVector = pmap(
            #     # distribute director problems to managers
            #     managerProblemVector -> pmap(
            #         # distributed manager problems to workers
            #         workerProblemVector -> gpu(
            #             workerProblemVector,
            #             finePropagator
            #         ),
            #         intersect(procs(myid()), DEVPOOL), # worker ids for that manager
            #         Iterators.partition(managerProblemVector, batch_size) .|> collect
            #     )
            #     # if I'm the director, distribute over the managers
            #     # if I'm a manager, distribute over the device processes on this machine
            #     CachingPool(MANAGERPOOL),
            #     Iterators.partition(directorProblemVector, batch_size) .|> collect
            # )
            batched_director_problems = batchProblems(directorProblemVector, MANAGERPOOL)
            # println("Distributing $(length(batched_director_problems)) problem sets from ", myid(), " to ", MANAGERPOOL)
            manager_solutions = pmap(CachingPool(MANAGERPOOL), batched_director_problems) do managerProblemVector
                
                my_worker_pool           = intersect(procs(myid()), DEVPOOL)
                batched_manager_problems = batchProblems(managerProblemVector, my_worker_pool)
                # println("Distributing $(length(batched_manager_problems)) problem sets from ", myid(), " to ", my_worker_pool)
                worker_solutions = pmap(
                    workerProblemVector -> gpu(workerProblemVector, finePropagator),
                    CachingPool(my_worker_pool),
                    batched_manager_problems
                )
                return vcat(worker_solutions...)
            end
            subSolutionFineVector = vcat(manager_solutions...)
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
        newSolution = rootSolution
        has_converged, maxPositionPercentChange, maxVelocityPercentChange = hasConverged(oldSolution, newSolution; threshold)
        oldSolution = newSolution
        print(
            "Finished: log10(max(|%Δposition|)) = ", 
            round(Int, maxPositionPercentChange |> log10), 
            ", log10(max(|%Δvelocity|)) = ", 
            round(Int, maxVelocityPercentChange |> log10), 
            "\r"
        )
        # create new sub problems
        if iteration != initialDiscretization # no need for new subproblems after last iteration
            # println("Updating subproblems")
            updateSubproblems!(
                directorProblemVector, 
                rootSolution, 
                ivp.acceleration
            )
        end
    end
# ==================================================================================================
    if iteration != maxIterations
        printstyled("\nConvergence achieved. ", color=:green)
    else
        printstyled("\nFailed to converge in $maxIterations iterations. ", color=:yellow)
    end
    return rootSolution
end

function solve(
    ivp :: SecondOrderIVP,
    coarse :: Propagator, 
    fine :: Propagator;
    threshold = 10^(-10)
) :: Solution
    println("Beginning parareal evaluation")
    sol = parareal(ivp, coarse, fine; threshold = threshold)
    println("Closing cluster.")
    # TODO: print stats e.g. total problems, iterations, steps
    rmprocs(workers())
    return sol
end