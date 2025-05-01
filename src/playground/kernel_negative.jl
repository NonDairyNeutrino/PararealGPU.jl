using CUDA

kernel!(x_dev) = (-x_dev; nothing)

x = [1,2,3]
x_dev = cu(x)

# works just fine
-x_dev

# Does not work
@cuda kernel!(x_dev)
