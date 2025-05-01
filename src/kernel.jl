"""
    kernelPrep(subProblemVector :: Vector{SecondOrderIVP}, discretization :: Int) :: Tuple{Matrix, Array, Array}

Initializes host-based arrays for the discretized domains, positions, and velocities.
"""
function kernelPrep(subProblemVector :: Vector{SecondOrderIVP}, discretization :: Int) :: Tuple{Matrix, Array, Array}
    solutionCount     = length(subProblemVector)
    sequenceLength    = discretization
    positionDimension = subProblemVector[1].initialPosition |> length

    # port domain bounds to an array, THEN put the whole thing on the device
    discretizedDomain = zeros(sequenceLength, solutionCount)
    for (i, problem) in enumerate(subProblemVector)
        discretizedDomain[1, i]   = problem.domain.lb
        discretizedDomain[end, i] = problem.domain.ub
    end

    # where to put stuff
    # arrays should be indexed such that elements are in columns for performance
    position          = zeros(solutionCount, positionDimension, sequenceLength)
    position[:, :, 1] = getproperty.(subProblemVector, :initialPosition) |> stack

    velocity          = zeros(solutionCount, positionDimension, sequenceLength)
    velocity[:, :, 1] = getproperty.(subProblemVector, :initialVelocity) |> stack

    return discretizedDomain, position, velocity
end

# """
#     discretizeKernel!(domainPointVector :: S, step :: T) where {S, T}

# Fill discretized domain with middle elements.
# """
# function discretizeKernel!(domainPointVector :: S, step :: T) where {S, T}
#     # why did I call it "domainPointVector"? Cause it's a vector of the points in the domain
#     # is that a particularly intuitive name? I don't know.
#     # what I do know is that domainPointVector takes the form 
#     # [lower bound, 0.0, 0.0, ... , 0.0, upper bound]
#     # why is it like that? So the gpu can purely just calculate and assign to an array index
#     # and not have to do any allocations
#     discretization = length(domainPointVector)
#     @inbounds lowerBound     = domainPointVector[begin]
#     i = 2
#     while i <= (discretization - 1) # for i in 2:(discretization - 1)
#         @inbounds domainPointVector[i] = lowerBound + (i - 1) * step
#         i += 1
#     end
#     return nothing
# end

# """
#     propagateKernel!(solver :: F, acceleration :: G, step :: H, positionSequence :: J, velocitySequence :: K) :: Nothing where {F, G, H, J, K}

# Propagates on the device.
# """
# function propagateKernel!(
#     solver           :: F, 
#     acceleration     :: G, 
#     step             :: H, 
#     positionSequence :: J, 
#     velocitySequence :: K
#     ) :: Nothing where {F, G, H, J, K}
#     discretization = size(positionSequence, 2) # number of positions in the sequence
#     @cuprintln("fine discretization = ", discretization)

#     i = 2
#     @views while i <= discretization # for i in 2:discretization
#         @inbounds oldPosition = positionSequence[:, i - 1]
#         @inbounds oldVelocity = velocitySequence[:, i - 1]
#         # @cushow oldVelocity
#         newPosition, newVelocity = solver(oldPosition, oldVelocity, acceleration, step)
#         # @cushow newVelocity
#         @inbounds positionSequence[:, i] .= newPosition
#         @inbounds velocitySequence[:, i] .= newVelocity
#         i += 1
#     end
#     return nothing
# end

# """
#     kernel!(solver, acceleration, discretizedDomain, position, velocity) :: Nothing

# CUDA kernel to propagate the subProblems.
# """
# function kernel!(solver, acceleration, discretizedDomain, position, velocity) :: Nothing
#     discretization, solutionCount = size(discretizedDomain)
#     index         = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride        = gridDim().x * blockDim().x
#     i = index
#     @views while i <= solutionCount # for i = index:stride:solutionCount
#         domainPointVector = discretizedDomain[:, i]
#         lowerBound        = domainPointVector[begin]
#         upperBound        = domainPointVector[end]
#         step              = (upperBound - lowerBound) / discretization

#         # indexing follows [# of problems, vector dimension, FINEDISCRETIZATION]
#         positionSequence = position[i, :, :]
#         velocitySequence = velocity[i, :, :]

#         @cuprintln("discretizing")
#         # these kernel calls also execute asynchronously!
#         #= CUDA.@sync =# discretizeKernel!(domainPointVector, step)
#         @cuprintln("propagating")
#         #= CUDA.@sync =# propagateKernel!(solver, acceleration, step, positionSequence, velocitySequence)
#         @cuprintln("done propagating")
#         # synchronize()
#         i += stride
#     end
#     return nothing
# end

"""
    propagate_gpu!(
        problemCount :: Int,
        dimension    :: Int,
        t_max        :: Int,
        step         :: Float32,
        pos_seqs_dev,
        vel_seqs_dev
    ) :: Nothing

TBW
"""
function propagate_gpu!(
        problemCount :: Int,
        dimension    :: Int,
        t_max        :: Int,
        step         :: Float32,
        pos_seqs_dev,
        vel_seqs_dev
    ) :: Nothing 
    problem      = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride       = gridDim().x * blockDim().x

    # problemCount, dimension, t_max = size(pos_seqs_dev)

    while problem <= problemCount
        t = 2 # t = 1 is the initialvalues, which are already populated in the arrays
        while t <= t_max
            d = 1
            while d <= dimension
                x_old = @inbounds pos_seqs_dev[problem, dimension, t - 1]
                v_old = @inbounds vel_seqs_dev[problem, dimension, t - 1]

                acc_old = -x_old
                x_new   = x_old + v_old * step + 0.5 * acc_old * step^2
                acc_new = -x_new
                v_new   = v_old + 0.5 * (acc_old + acc_new) * step

                @inbounds pos_seqs_dev[problem, dimension, t - 1] = x_new
                @inbounds vel_seqs_dev[problem, dimension, t - 1] = v_new
                d += 1
            end

            @cuprintln("Calculated values at time $t on problem $problem on thread ", threadIdx().x)
            t += 1
        end
        problem += stride
    end
    return nothing
end

"""
    pararealSolution!(
        solver            :: F, 
        acceleration      :: G, 
        discretizedDomain :: Matrix{T}, 
        pos_seqs          :: Array{T, 3}, 
        vel_seqs          :: Array{T, 3}
    ) :: Vector{Solution} where {T <: AbstractFloat, F <: Function, G<: Function}

Optimize hardware usage and execute the kernel.
"""
function pararealSolution!(
        int_scheme        :: F, 
        acceleration      :: G, 
        discretizedDomain :: Matrix{T}, 
        pos_seqs          :: Array{T, 3}, 
        vel_seqs          :: Array{T, 3}
    ) :: Vector{Solution} where {T <: AbstractFloat, F <: Function, G <: Function}

    # the domains are easy enough to just make on the host
    # done in-place
    discretizedDomain .= mapslices(
        dom -> range(first(dom), last(dom), length(dom)), 
        discretizedDomain, 
        dims = 1
    )
    # all problems use same step size
    step = (discretizedDomain[end, 1] - discretizedDomain[begin,1]) / size(discretizedDomain, 1) |> Float32

    # put stuff on the gpu
    problemCount, dimension, t_max = size(pos_seqs)
    pos_seqs_dev = pos_seqs |> cu
    vel_seqs_dev = vel_seqs |> cu
    println("Arrays copied to device.")

    # solver(scheme, acc, step) = ((x, v) -> scheme(x, v, acc, step))
    # foo = solver(int_scheme, acceleration, step)

    # optimize the kernel parameters e.g. threads, blocks
    kernel_call = @cuda launch=false propagate_gpu!(
        problemCount,
        dimension,
        t_max,
        step,
        pos_seqs_dev, 
        vel_seqs_dev
    )
    printstyled("Kernel successfully compiled\n", color=:green)
    config  = launch_configuration(kernel_call.fun)
    threads = min(problemCount, config.threads)
    blocks  = cld(problemCount, threads)
    println("Evaluating on $blocks blocks and $threads threads per block.")

    # execute on the gpu
    @show vel_seqs[1, :, :]
    CUDA.@sync kernel_call(
        problemCount,
        dimension,
        t_max,
        step,
        pos_seqs_dev, 
        vel_seqs_dev,
        threads,
        blocks
    )
    printstyled("Kernel has finished!\n", color=:green)

    # pull off the gpu
    pos_seqs .= mapslices(p -> vec.(eachcol(p)), pos_seqs_dev |> Array; dims=1)
    vel_seqs .= mapslices(p -> vec.(eachcol(p)), vel_seqs_dev |> Array; dims=1)
    @show vel_seqs[1, :, :]

    solutionVector = Vector{Solution}(undef, problemCount)
    for i in eachindex(solutionVector)
        solutionVector[i] = Solution(discretizedDomain[:, i], pos_seqs[i, :, :], vel_seqs[i, :, :])
    end

    return solutionVector
end