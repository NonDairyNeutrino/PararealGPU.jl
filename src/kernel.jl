"""
    kernelPrep(
        subProblemVector :: Vector{SecondOrderIVP{T}}, 
        discretization :: Int
    ) :: Tuple{Matrix{T}, Array{T, 3}, Array{T, 3}} where T <: AbstractFloat

Initializes host-based arrays for the discretized domains, positions, and velocities.
"""
function kernelPrep(
        subProblemVector :: Vector{SecondOrderIVP{T}}, 
        discretization :: Int
    ) :: Tuple{Matrix{T}, Array{T, 3}, Array{T, 3}} where T <: AbstractFloat
    solutionCount = length(subProblemVector)
    t_max         = discretization
    dimension     = subProblemVector[1].initialPosition |> length

    # port domain bounds to an array, THEN put the whole thing on the device
    discretizedDomain = Matrix{T}(undef, t_max, solutionCount)
    Threads.@threads for (i, problem) in collect(enumerate(subProblemVector))
        discretizedDomain[begin, i] = problem.domain.lb
        discretizedDomain[end,   i] = problem.domain.ub
    end

    # where to put stuff
    # arrays should be indexed such that elements are in columns for performance
    # 2 comes from the fact that these arrays are only storing the initial and final values
    position          = Array{T, 3}(undef, solutionCount, dimension, 2)
    position[:, :, 1] = getproperty.(subProblemVector, :initialPosition) |> stack

    velocity          = Array{T, 3}(undef, solutionCount, dimension, 2)
    velocity[:, :, 1] = getproperty.(subProblemVector, :initialVelocity) |> stack

    return discretizedDomain, position, velocity
end

function kernelPrep!(
        subProblemVector :: Vector{SecondOrderIVP{T}},
        timeMatrix       :: Matrix{T},
        positionArray    :: Array{T, 3},
        velocityArray    :: Array{T, 3}
    ) :: Nothing where T <: AbstractFloat

    timeMatrix[begin,   :] .= (p -> p.domain.lb).(subProblemVector)
    timeMatrix[end,     :] .= (p -> p.domain.ub).(subProblemVector)

    positionArray[:, :, 1] .= getproperty.(subProblemVector, :initialPosition) |> stack |> permutedims
    velocityArray[:, :, 1] .= getproperty.(subProblemVector, :initialVelocity) |> stack |> permutedims
    return nothing
end

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
    # FIXME: FOR SOME REAOSN GOD ONLY KNOWS, THE AMPLITUDE OF THE POSITION WAVE IS CONSTANT IN TIME
    # BUT DEPENDS ON THE WAVE WAVE NUMBER I.E. IT DOES NOT DECAY IN TIME BUT DECREASES WITH A HIGHER
    # FREQUENCY
    k            = 1.0f0 # * pi
    k2           = k^2
    halfstep     = 0.5f0 * step
    halfstep2    = halfstep * step
    acc(r)       = -k2 * r

    while problem <= problemCount
        dim = 1
        while dim <= dimension
            pos = @inbounds pos_seqs_dev[problem, dim, 1]
            vel = @inbounds vel_seqs_dev[problem, dim, 1]

            t = 1
            while t <= t_max
                acc_old = acc(pos)
                pos    += vel*step + halfstep2*acc_old
                acc_new = acc(pos)
                vel    += halfstep * (acc_old + acc_new)
                t += 1
            end

            # @cuassert isfinite(pos) "final position is NaN or infinite"
            @inbounds pos_seqs_dev[problem, dim, 2] = pos
            # @cuassert isfinite(vel) "final velocity is NaN or infinite"
            @inbounds vel_seqs_dev[problem, dim, 2] = vel

            dim += 1
        end
        # @cuprintln("From GPU block.thread $(blockIdx().x).$(threadIdx().x): problem $problem has been calculated.")
        problem += stride
    end
    return nothing
end

"""
    pararealSolution!(
        int_scheme        :: F, 
        acceleration      :: G, 
        discretizedDomain :: Matrix{T}, 
        pos_seqs          :: Array{T, 3}, 
        vel_seqs          :: Array{T, 3}
    ) :: Vector{Solution} where {T <: AbstractFloat, F <: Function, G <: Function}

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
    step = (discretizedDomain[end, 1] - discretizedDomain[begin,1]) / size(discretizedDomain, 1)
    # @info "dt/T = $(1/size(discretizedDomain, 1))" maxlog=1

    # put stuff on the gpu
    t_max        = size(discretizedDomain, 1)
    problemCount, dimension, _ = size(pos_seqs)
    pos_seqs_dev = pos_seqs |> CuArray
    vel_seqs_dev = vel_seqs |> CuArray
    # println("Arrays copied to device.")

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
    # printstyled("Kernel successfully compiled\n", color=:green)
    config  = launch_configuration(kernel_call.fun)
    threads = min(problemCount, config.threads)
    blocks  = cld(problemCount, threads)
    @info string(
        "Evaluating $problemCount problems on $blocks blocks and $threads threads per block for ", 
        round(problemCount / (blocks * threads), sigdigits=2), 
        " problems per thread"
    ) maxlog=1

    # execute on the gpu
    try
        CUDA.@sync kernel_call(
            problemCount,
            dimension,
            t_max,
            step,
            pos_seqs_dev, 
            vel_seqs_dev; # this needs to be a semicolon ";" and not a comma ","
            threads,      # or else it will run single-threaded for some godforsaken reason
            blocks
        )
    catch e
        println(
            """
            An error occurred on the GPU.  This is probably due to a big step size leading to NaNs.\
            Consider increasing either of the discretizations or making the domain smaller.\
            This will be addressed in the future.\
            """
        )
        rethrow()
    end
    # printstyled("Kernel has finished!\n", color=:green)

    # pull off the gpu
    pos_seqs .= Array(pos_seqs_dev)
    @assert all(isfinite, pos_seqs) "$(count(!isfinite, pos_seqs)) infs or nans in pos_seqs.  First at $(findfirst(!isfinite, pos_seqs))"
    vel_seqs .= Array(vel_seqs_dev)

    solutionVector = Vector{Solution}(undef, problemCount)
    Threads.@threads for problem in eachindex(solutionVector)
        dom     = discretizedDomain[:, problem]
        pos_seq = collect.(eachcol(pos_seqs[problem, :, :]))
        vel_seq = collect.(eachcol(vel_seqs[problem, :, :]))
        solutionVector[problem] = Solution(dom, pos_seq, vel_seq)
    end

    return solutionVector
end
