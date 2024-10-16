"""
    pararealKernel(subProblems, propagator, subSolution)

TBW
"""
function pararealKernel(subProblems, propagator, subSolution)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    # could change below to while loop to avoid StepRangeLength for performance
    for i = index:stride:length(subProblems)
        @inbounds subSolution[i] = propagate(subProblems[i], propagator)
    end
    return nothing
end