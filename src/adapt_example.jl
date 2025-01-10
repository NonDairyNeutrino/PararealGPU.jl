using CUDA, Adapt

struct TestStruct{A}
    testField :: A
end

Adapt.@adapt_structure TestStruct

function squareGPU!(test)
    testField = test.testField
    index     = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride    = gridDim().x * blockDim().x
    for i in index:stride:length(testField)
        testField[i] = testField[i]^2
    end
    return nothing
end

const N = 10^9
test = TestStruct(CUDA.rand(N))

kernel  = @cuda launch=false squareGPU!(test)
config  = launch_configuration(kernel.fun)
threads = min(N, config.threads)
blocks  = cld(N, threads)
println("executing $N squares on $(blocks) blocks and $(threads) threads per block")
display(CUDA.@bprofile kernel(test; threads, blocks))
