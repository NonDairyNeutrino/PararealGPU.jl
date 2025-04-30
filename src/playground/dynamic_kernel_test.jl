using CUDA

threadCount = 10

function innerKernel(outerThread)
    innerThread = threadIdx().x
    for i in 1:10
        @cuprintln("outer: $outerThread, inner: $innerThread, iteration: $i")
    end
    return
end

function outerKernelStatic()
    outerThread = threadIdx().x
    innerKernel(outerThread)
    return
end

function outerKernelDynamic(threadCount :: Int)
    outerThread = threadIdx().x
    @cuda dynamic=true threads=threadCount innerKernel(outerThread)
    return
end

println("Executing static kernel")
CUDA.@sync @cuda threads=threadCount outerKernelStatic()

# println("Executing dynamic kernel")
# CUDA.@sync @cuda threads=threadCount outerKernelDynamic(threadCount)