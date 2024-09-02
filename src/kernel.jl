"""
    Adapt.adapt_structure(to, itp::Interpolate)

TBW
"""
function Adapt.adapt_structure(to, itp::Interpolate)
    xs = Adapt.adapt_structure(to, itp.xs)
    ys = Adapt.adapt_structure(to, itp.ys)
    Interpolate(xs, ys)
end

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