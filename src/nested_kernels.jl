# kernel_test.jl
# can you use a function inside a kernel?
using CUDA

"""
    innerKernel(x :: T) where T

Shared function between the host and device.
"""
function innerKernel!(xVector :: T) where T
    xVector .= xVector.^2
end

"""
    outerKernel!(xVector)

Function that runs on the device.
"""
function outerKernel!(xVector)
    innerKernel!(xVector)
    return nothing
end

argc = 2^10
cudaVector = CUDA.rand(argc)

@cuda threads=argc outerKernel!(cudaVector)
display(cudaVector)
