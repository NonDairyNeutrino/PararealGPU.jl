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

# TODO: Adapt.@adapt_structure the following structures
# - SecondOrderIVP
# - Propagator
# - Solution
