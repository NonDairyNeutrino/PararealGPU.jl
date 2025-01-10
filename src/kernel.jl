using CUDA
"""
    pararealKernel(subProblems, propagator, subSolution)

TBW
"""
function pararealKernel(subProblemVector, propagator, subSolutionVector)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    # could change below to while loop to avoid StepRangeLength for performance
    for i = index:stride:length(subProblemVector)
        @inbounds subSolutionVector[i] = propagate(subProblemVector[i], propagator)
    end
    return nothing
end

"""
    kernelPrep(subProblemVector :: Vector{SecondOrderIVP}, discretization :: Real) :: Vector{Matrix}

Initializes host-based arrays for the discretized domains, positions, and velocities.
"""
function kernelPrep(subProblemVector :: Vector{SecondOrderIVP}, discretization :: Real) :: Tuple{Matrix, Matrix, Matrix}
    solutionCount            = length(subProblemVector)
    sequenceLength     = discretization
    positionDimension = 3

    # port domain bounds to an array, THEN put the whole thing on the device
    discretizedDomain = zeros(sequenceLength, solutionCount)
    for (i, problem) in enumerate(subProblemVector)
        discretizedDomain[1, i]   = problem.domain.lb
        discretizedDomain[end, i] = problem.domain.ub
    end

    # where to put stuff
    # arrays should be indexed such that elements are in columns for performance
    position          = zeros(solutionCount, positionDimension, sequenceLength)
    position[:, :, 1] = getproperty.(subProblemVector, :initialPosition)

    velocity          = zeros(solutionCount, positionDimension, sequenceLength)
    velocity[:, :, 1] = getproperty.(subProblemVector, :initialVelocity)

    return discretizedDomain, position, velocity
end

"""
    kernelWrapper(solver, acceleration, discretizedDomain, position, velocity)

Optimize hardware usage and execute the kernel.
"""
function kernelWrapper(solver, acceleration, discretizedDomain, position, velocity) :: Tuple
    problemCount      = size(discretizedDomain, 2)
    discretizedDomain = discretizedDomain |> cu
    position          = position          |> cu
    velocity          = velocity          |> cu
    
    kernel  = @cuda launch=false kernel!(solver, acceleration, discretizedDomain, position, velocity)
    config  = launch_configuration(kernel.fun)
    threads = min(problemCount, config.threads)
    blocks  = cld(problemCount, threads)
    # actually launches the kernel on the device
    kernel(solver, acceleration, discretizedDomain, position, velocity; threads, blocks)
    return discretizedDomain, position, velocity
end