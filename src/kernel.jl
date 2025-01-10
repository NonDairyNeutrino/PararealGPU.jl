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
    kernelPrep(subProblemVector :: Vector{SecondOrderIVP}, discretization :: Real) :: Vector{CuArray}

Initializes device arrays for the discretized domains, positions, and velocities.
"""
function kernelPrep(subProblemVector :: Vector{SecondOrderIVP}, discretization :: Real) :: Vector{CuArray}
    solutionCount            = length(subProblemVector)
    sequenceLength     = discretization
    positionDimension = 3

    # port domain bounds to an array, THEN put the whole thing on the device
    discretizedDomain = zeros(sequenceLength, solutionCount)
    for (i, problem) in enumerate(subProblemVector)
        discretizedDomain[1, i]   = problem.domain.lb
        discretizedDomain[end, i] = problem.domain.ub
    end
    discretizedDomain = discretizedDomain |> cu

    # where to put stuff
    # arrays should be indexed such that elements are in columns for performance
    position          = zeros(solutionCount, positionDimension, sequenceLength)
    position[:, :, 1] = getproperty.(subProblemVector, :initialPosition)
    position          = position |> cu

    velocity          = zeros(solutionCount, positionDimension, sequenceLength)
    velocity[:, :, 1] = getproperty.(subProblemVector, :initialVelocity)
    velocity          = velocity |> cu

    return discretizedDomain, position, velocity
end
