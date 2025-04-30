# https://discourse.julialang.org/t/calling-a-function-inside-of-a-kernel/25023?u=nondairyneutrino

using CUDA

cloo(foo, a) = ((x,y) -> foo(x, y) + a)

function kernel!(clib, fib_array)
    i = 3
    while i <= length(fib_array)
        @cuprintln("Calculating step $i")
        fib_array[i] = clib(fib_array[i-2], fib_array[i-1])
        i += 1
    end
    return nothing
end

max_steps = 10
fib_array = Vector{Float32}(undef, max_steps)
fib_array[1] = 1.0f0
fib_array[2] = 1.0f0
fib_array_dev = fib_array |> cu
println("Array copied from host to device.")

fib(x, y) = x + y
a = 1.0f0
clib = cloo(fib, a)

# kernel_call = @cuda launch=false kernel!(clib, fib_array_dev)
# printstyled("Kernel successfully compiled\n", color=:green)
# config  = launch_configuration(kernel_call.fun)
# threads = min(max_steps, config.threads)
# blocks  = cld(max_steps, threads)
# println("Evaluating on $blocks blocks and $threads threads per block.")

println("Launching kernel on the device")
# CUDA.@sync @cuda kernel_call(clib, fib_array_dev, threads, blocks)
CUDA.@sync @cuda threads=max_steps kernel!(clib, fib_array_dev)
println("Kernel finished.")

fib_array .= fib_array_dev |> Array
println("Array copied from device to host. Displaying...")
display(fib_array)
