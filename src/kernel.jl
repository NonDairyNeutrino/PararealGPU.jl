"""
    kernelPrep(subProblemVector :: Vector{SecondOrderIVP}, discretization :: Int) :: Tuple{Matrix, Array, Array}

Initializes host-based arrays for the discretized domains, positions, and velocities.
"""
function kernelPrep(subProblemVector :: Vector{SecondOrderIVP}, discretization :: Int) :: Tuple{Matrix, Array, Array}
    solutionCount            = length(subProblemVector)
    sequenceLength     = discretization
    positionDimension = subProblemVector[1].initialPosition |> length

    # port domain bounds to an array, THEN put the whole thing on the device
    discretizedDomain = zeros(sequenceLength, solutionCount)
    for (i, problem) in enumerate(subProblemVector)
        discretizedDomain[1, i]   = problem.domain.lb
        discretizedDomain[end, i] = problem.domain.ub
    end

    # where to put stuff
    # arrays should be indexed such that elements are in columns for performance
    position          = zeros(solutionCount, positionDimension, sequenceLength)
    position[:, :, 1] = getproperty.(subProblemVector, :initialPosition) |> stack

    velocity          = zeros(solutionCount, positionDimension, sequenceLength)
    velocity[:, :, 1] = getproperty.(subProblemVector, :initialVelocity) |> stack

    return discretizedDomain, position, velocity
end

"""
    discretizeKernel!(domainPointVector :: T) where T

Fill discretized domain with middle elements.
"""
function discretizeKernel!(domainPointVector :: S, step :: T) where {S, T}
    discretization = length(domainPointVector)
    lowerBound     = domainPointVector[begin]
    for i in 2:(discretization - 1)
        @inbounds domainPointVector[i] = lowerBound + (i - 1) * step
    end
    return nothing
end

"""
    propagateKernel!(solver :: F, acceleration :: G, step :: H, positionSequence :: J, velocitySequence :: K) :: Nothing where {F, G, H, J, K}

Propagates on the device.
"""
function propagateKernel!(solver :: F, acceleration :: G, step :: H, positionSequence :: J, velocitySequence :: K) :: Nothing where {F, G, H, J, K}
    discretization = size(positionSequence, 3) # number of positions in the sequence
    @views for i in 2:(discretization - 1)
        oldPosition = positionSequence[:, i - 1]
        oldVelocity = velocitySequence[:, i - 1]
        newPosition = positionSequence[:, i]
        newVelocity = velocitySequence[:, i]
        @inbounds newPosition, newVelocity = solver(oldPosition, oldVelocity, acceleration, step)
    end
    return nothing
end

"""
    kernel!(solver, acceleration, discretizedDomain, position, velocity) :: Nothing

CUDA kernel to propagate the subProblems.
"""
function kernel!(solver, acceleration, discretizedDomain, position, velocity) :: Nothing
    discretization, solutionCount = size(discretizedDomain)
    index         = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride        = gridDim().x * blockDim().x
    # could change below to while loop to avoid StepRangeLength for performance
    @views for i = index:stride:solutionCount
        domainPointVector = discretizedDomain[:, i]
        lowerBound        = domainPointVector[begin]
        upperBound        = domainPointVector[end]
        step              = (upperBound - lowerBound) / discretization

        positionSequence = position[i, :, :]
        velocitySequence = velocity[i, :, :]

        discretizeKernel!(domainPointVector, step)
        propagateKernel!(solver, acceleration, domainPointVector, positionSequence, velocitySequence)
    end
    return nothing
end

Optimize hardware usage and execute the kernel.
"""
function pararealSolution(solver, acceleration, discretizedDomain, position, velocity) :: Vector{Solution}
    problemCount      = size(discretizedDomain, 2)
    discretizedDomain = discretizedDomain |> cu
    position          = position          |> cu
    velocity          = velocity          |> cu
    
    kernel  = @cuda launch=false kernel!(solver, acceleration, discretizedDomain, position, velocity)
    config  = launch_configuration(kernel.fun)
    threads = min(problemCount, config.threads)
    blocks  = cld(problemCount, threads)
    # actually launches the kernel on the device
    println("Evaluating on $blocks blocks and $threads threads per block.")
    kernel(solver, acceleration, discretizedDomain, position, velocity; threads, blocks)
    println("GPU execution successful!")

    subSolutionVector = Vector{Solution}(undef, problemCount)
    for i in 1:problemCount
        subSolutionVector[i] = Solution(discretizedDomain[:, i], position[i, :, :], velocity[i, :, :])
    end

    return subSolutionVector
end