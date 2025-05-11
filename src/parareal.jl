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
    problemVector  :: Vector{SecondOrderIVP{T}},
    prop           :: Propagator
    # solutionVector :: Vector{Solution}
    ) :: Vector{Solution{T}} where T <: AbstractFloat
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
        discretizedDomain, 
        positionMatrix, 
        velocityMatrix
    )
    @info "Parareal evaluation complete.  Sending $(round(sizeof(solutionVector) / 1000^2, sigdigits=2)) MB vector of solutions." maxlog=1
    return solutionVector
end

"""
    batchProblems(problemVector, my_worker_pool)

TBW
"""
function batchProblems(problemVector, my_worker_pool)
    batch_size = div(length(problemVector), length(my_worker_pool))
    # if problemVector does not divide evenly into MANAGERPOOL
    # Iterators.partition includes a "remainder" partition; see docs for Iterators.partition
    return Iterators.partition(problemVector, batch_size) .|> collect
end

"""
    parareal(
        ivp              :: SecondOrderIVP{T}, 
        coarsePropagator :: Propagator, 
        finePropagator   :: Propagator;
        threshold = convert(T, 1.0e-10)
    ) :: Solution{T} where T <: AbstractFloat

Solve the given IVP using the distributed, GPU-based Parareal algorithm.
"""
function parareal(
        ivp              :: SecondOrderIVP{T}, 
        coarsePropagator :: Propagator, 
        finePropagator   :: Propagator;
        threshold        :: T = convert(T, 1.0e-10),
        localonly        :: Bool = false
    ) :: Solution{T} where T <: AbstractFloat
# ==================================================================================================
    # INITIALIZATION
    @info "Beginning iteration 0"
    rootSolution, directorProblemVector = initializeSubproblems(ivp, coarsePropagator)
    predSolution = rootSolution # on iteration 0 predicted solution = root solution
    oldSolution  = rootSolution

    initialDiscretization   = coarsePropagator.discretization
    subSolutionCoarseVector = similar(directorProblemVector, Solution{T})
    subSolutionFineVector   = similar(directorProblemVector, Solution{T})
    positionCorrectorVector = similar(directorProblemVector, Vector{T})
    velocityCorrectorVector = similar(directorProblemVector, Vector{T})

    # pre-allocate arrays for if localonly
    # can't put in if block for scope
    problemCount  = length(directorProblemVector)
    t_max         = finePropagator.discretization
    dimension     = ivp.initialPosition |> first |> length
    timeMatrix    = Matrix{T}(undef, t_max, problemCount)
    positionArray = Array{T, 3}(undef, problemCount, dimension, 2) # 2 for only the initial and final values
    velocityArray = Array{T, 3}(undef, problemCount, dimension, 2)

    iteration        = 0
    maxIterations    = coarsePropagator.discretization
# ==================================================================================================
    # BEGIN LOOP
    has_converged = false
    percentage_iterations = maxIterations / 100 > 1 ? floor.(Int, (1:100) * (maxIterations / 100)) : Base.OneTo(100)
    while !has_converged && iteration < maxIterations 
        iteration += 1
# ==================================================================================================
        # offload and propagate in parallel
        # batched_director_problems = batchProblems(directorProblemVector, MANAGERPOOL)
        # manager_solutions = pmap(CachingPool(MANAGERPOOL), batched_director_problems) do managerProblemVector
        #     my_worker_pool           = intersect(procs(myid()), DEVPOOL)
        #     batched_manager_problems = batchProblems(managerProblemVector, my_worker_pool)
        #     # println("Distributing $(length(batched_manager_problems)) problem sets from ", myid(), " to ", my_worker_pool)
        #     worker_solutions = pmap(
        #         workerProblemVector -> gpu(workerProblemVector, finePropagator),
        #         CachingPool(my_worker_pool),
        #         batched_manager_problems
        #     )
        #     return vcat(worker_solutions...)
        # end
        # subSolutionFineVector = vcat(manager_solutions...)

        # go straight to gpu, do not pass manager
        if localonly
            kernelPrep!(directorProblemVector, timeMatrix, positionArray, velocityArray)
            subSolutionFineVector .= pararealSolution!(
                finePropagator.propagator,
                ivp.acceleration, 
                timeMatrix, 
                positionArray, 
                velocityArray
            )
        else
            batched_worker_problems = batchProblems(directorProblemVector, DEVPOOL)
            @info "$(sizeof(first(batched_worker_problems)) / 1000^2) MB of problems will be sent to each of the $(length(DEVPOOL)) workers each iteration." maxlog=1
            subSolutionFineVector .= pmap(
                workerProblemVector -> gpu(workerProblemVector, finePropagator), 
                CachingPool(DEVPOOL), 
                batched_worker_problems
            ) |> (vv -> vcat(vv...))
        end
# ==================================================================================================
        # correction
        # use the values given by the coarse propagator from the previous iteration
        Threads.@threads for (i, (pos, vel)) in collect(enumerate(zip(predSolution.positionSequence[2:end], predSolution.velocitySequence[2:end])))
            # correct! only needs the "last" value in the sequence, so just wrap the value in the an array
            # this emulates the coarse propagator also running for each subproblem because the
            # coarse propagator only takes one step
            # the rest of th
            subSolutionCoarseVector[i] = Solution(T[], [pos], [vel])
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
        if #= iteration in percentage_iterations && =# !iszero(maxPositionPercentChange) && !iszero(maxVelocityPercentChange)
            print(
                "Finished iteration $iteration: log10(max(|%Δposition|)) = ", 
                round(Int, maxPositionPercentChange |> log10),
                ", log10(max(|%Δvelocity|)) = ", 
                round(Int, maxVelocityPercentChange |> log10),
                "\n"
            )
        end
        # create new sub problems
        if iteration != initialDiscretization && !has_converged  # no need for new subproblems after last iteration
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
        printstyled("\nConvergence achieved in $iteration iterations.\n", color=:green)
    else
        printstyled("\nFailed to converge in $maxIterations iterations.\n", color=:yellow)
    end
    return rootSolution
end

"""
    solve(
        nodeVector           :: Vector{String},
        coarseIntegrator     :: Function,
        coarseDiscretization :: Int,
        fineIntegrator       :: Function,
        fineDiscretization   :: Int,
        acc                  :: Function,
        lowerBound           :: T,
        upperBound           :: T,
        initialPosition      :: Vector{T},
        initialVelocity      :: Vector{T};
        addlocal             :: Bool = false,
        threshold            :: T    = convert(T, 10)
    ) :: Solution{T} where T <: AbstractFloat

Solves the given problem using the given coarse and fine propagators.
"""
function solve(
        nodeVector           :: Vector{String},
        coarseIntegrator     :: Function,
        coarseDiscretization :: Int,
        fineIntegrator       :: Function,
        fineDiscretization   :: Int,
        acc                  :: Function,
        lowerBound           :: T,
        upperBound           :: T,
        initialPosition      :: Vector{T},
        initialVelocity      :: Vector{T};
        addlocal             :: Bool = false,
        localonly            :: Bool = false,
        threshold            :: T    = convert(T, 10)
    ) :: Solution{T} where T <: AbstractFloat
    sizeof(T) > 4 && @warn "Floats are larger than 32 bits. Consider downsizing to increase GPU performance." T

    !localonly && prepCluster(nodeVector, addlocal = addlocal)
    @info "Creating initial value problems"
    coarse = Propagator(coarseIntegrator, coarseDiscretization)
    fine   = Propagator(fineIntegrator,  fineDiscretization)

    # redefine and distribute to allow the user to stipulate a generic function
    # but have the underlying implementation specialize on floats and vectors
    function acceleration(
            position :: Vector{T}, velocity :: Vector{T}
        ) :: Vector{T} where T <: AbstractFloat
        return acc(position, velocity)
    end
    domain = Interval{T}(lowerBound, upperBound)
    ivp    = SecondOrderIVP("0", domain, acceleration, initialPosition, initialVelocity)

    @info "Beginning parareal evaluation"
    @time "Parareal evaluation took " sol = parareal(ivp, coarse, fine; threshold = threshold, localonly = localonly)
    println("Closing cluster.")
    # TODO: write solutions to a file just in case something goes wrong after this
    !localonly && rmprocs(workers())
    return sol
end