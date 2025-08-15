# module PararealGPUBenchmarks
# export bench_single, bench_all_single, bench_gpu, bench_all_gpu, bench_distributed, bench_all_distributed

using BenchmarkTools, DelimitedFiles, Distributed, JLD2
const proj_dir = "../../"
include("$proj_dir/src/PararealGPU.jl"); using .PararealGPU

const DATADIR = dirname(@__DIR__) * "/data/"

function single(domain :: Vector{T}, init_pos :: Vector{T}, init_vel :: Vector{T}) :: Tuple{Solution{T}, Int} where T <: AbstractFloat
    pos_seq :: Vector{Vector{T}} = similar(domain, Vector{T})
    vel_seq :: Vector{Vector{T}} = similar(domain, Vector{T})

    time_step  = domain[2] - domain[1]
    pos_seq[1] = init_pos
    vel_seq[1] = init_vel
    for i in 2:length(domain)
        pos_seq[i], vel_seq[i] = FINEINTEGRATOR(pos_seq[i-1], vel_seq[i-1], (x, v) -> -WAVENUMBER * x, time_step)
    end
    return (Solution(domain, pos_seq, vel_seq), length(domain))
end

"""
    bench_single(coarse :: Int, fine :: Int) :: NamedTuple

TBW
"""
function bench_single(coarse :: Int) :: NamedTuple
    # the parareal algorithm converges to a solution that is theoretically identical to that which
    # be produced by using the fine integrator on the coarse discretization
    # thus to accurately compare the quality of result between the single threaded and parallel
    # implementations, the single threaded benchmarks should be done using the fine integrator on
    # the coarse discretization
    max_steps = coarse # * fine
    domain    = range(DOMAINLOWERBOUND, DOMAINUPPERBOUND, max_steps) |> collect
    bench     = @btimed single($domain, $INITIALPOSITION, $INITIALVELOCITY)
    return bench
end

"""
    bench_all_single(coarse_fine_matrix :: Matrix{Int}; file_name :: String = "") :: Matrix{Float64}

Benchmark all single threaded cases in parallel and return their results, optionally writing the results to a file.
"""
function bench_all_single(coarse_vector :: Vector{Int} #= coarse_fine_matrix :: Matrix{Tuple{Int, Int}} =#) :: Nothing

    cnt = Threads.Atomic{Int}(0)
    Threads.@threads for index in eachindex(coarse_vector)
        
        coarse = coarse_vector[index]
        println("Beginning single threaded benchmark with coarse = $coarse")
        bench_file = DATADIR * "single_c$(coarse).jld2"
        try
            # don't save bench to intermediate variable to prevent memory overflow
            save_object(bench_file, bench_single(2^coarse))

        catch e
            println("Caught error for coarse = $coarse.")
            println("Moving on to next discretization pair.")
            display(e)

        finally
            println("Finished ", Threads.atomic_add!(cnt, 1), "/", length(coarse_vector))
        end

    end

    return nothing
end

function bench_gpu(coarse :: Int, fine :: Int, params) :: NamedTuple
    bench = @btimed solve(
            $params.NODEVECTOR,
            $params.COARSEINTEGRATOR,
            $coarse,
            $params.FINEINTEGRATOR,
            $fine,
            $((r, v) -> -(1)^2 * r),
            $params.DOMAINLOWERBOUND,
            $params.DOMAINUPPERBOUND,
            $params.INITIALPOSITION,
            $params.INITIALVELOCITY;
            addlocal  = true,
            localonly = true
        )
    return bench
end

function bench_all_gpu(coarse_fine_matrix :: Matrix{Tuple{Int, Int}}, params) :: Nothing

    bench_file_name = DATADIR * "bench_gpu.jld2"
    bench_file      = jldopen(bench_file_name, "w")
    for index in eachindex(coarse_fine_matrix)

        coarse, fine = coarse_fine_matrix[index]
        println("Beginning gpu benchmark with coarse = $coarse, fine = $fine")

        try
            # don't save bench to intermediate variable to prevent memory overflow
            write(bench_file, "$coarse/$fine", bench_gpu(2^coarse, 2^fine, params))

        catch e
            println("Caught error for coarse = $coarse, fine = $fine.")
            println("Moving on to next discretization pair.")
            display(e)
            rethrow()

        finally
            println("Finished ", index, "/", length(coarse_fine_matrix))

        end
    end

    close(bench_file)
    return nothing
end

function bench_distributed(coarse_disc :: Int, fine_disc :: Int, ivp, params) :: NamedTuple
    coarse = PararealGPU.Propagator(params.COARSEINTEGRATOR, coarse_disc)
    fine   = PararealGPU.Propagator(params.FINEINTEGRATOR,  fine_disc)
    bench  = @btimed PararealGPU.parareal(
        $ivp, 
        $coarse, 
        $fine
    )
    return bench
end

function bench_all_distributed(coarse_vector :: Vector{Int}, fine_vector :: Vector{Int}, params) :: Nothing

    PararealGPU.prepCluster(params.NODEVECTOR, addlocal = true)
    ivp = PararealGPU.build_ivp(
        let k = params.WAVENUMBER; ((r, v) -> -k^2 * r) end, # use let to effectively interpolate wavenumber
        params.DOMAINLOWERBOUND, params.DOMAINUPPERBOUND, 
        params.INITIALPOSITION, params.INITIALVELOCITY
    )

    for coarse in coarse_vector
        bench_file_coarse = jldopen(DATADIR * "bench_dist_c$coarse.jld2", "w")
        for fine in fine_vector
            println("Beginning distributed benchmark with coarse = $coarse, fine = $fine")

            try
                # don't save bench to intermediate variable to prevent memory overflow
                write(bench_file_coarse, "$fine", bench_distributed(2^coarse, 2^fine, ivp, params))

            catch e
                println("Caught error for coarse = $coarse, fine = $fine.")
                println("Moving on to next discretization pair.")
                display(e)
                # rethrow()

            finally
                println("Finished coarse: ", coarse, " fine: ", fine)

            end
        end
        close(bench_file_coarse)
    end

    return nothing
end

function main(params) :: Nothing
    coarse_vector      = 3:14 |> collect
    fine_vector        = 3:14 |> collect
    coarse_fine_matrix = Iterators.product(coarse_vector, fine_vector) |> collect

    save_object(DATADIR * "disc_matrix.jld2", coarse_fine_matrix)
    # bench_all_single(coarse_vector)
    # bench_all_gpu(coarse_fine_matrix, params)
    bench_all_distributed(coarse_vector, fine_vector, params)

    return nothing
end

# end

# if this file is explicitly run, then actually do the benchmarks
# if abspath(PROGRAM_FILE) == @__FILE__
    # const proj_dir = "../../"
    # include("$proj_dir/src/PararealGPU.jl"); using .PararealGPU: symplecticEuler, velocityVerlet
    # using .PararealGPUBenchmarks

    const NODEVECTOR           = String["Electromagnetism"]
    const COARSEINTEGRATOR     = symplecticEuler
    const FINEINTEGRATOR       = velocityVerlet
    const WAVENUMBER           = 1.0f0 # * pi # DO NO CHANGE
    # ACCELERATION(r, v)         = -WAVENUMBER^2 * r         # simple harmonic oscillator
    const DOMAINLOWERBOUND     = 0.0f0
    const DOMAINUPPERBOUNDFACTOR = 10
    const DOMAINUPPERBOUND     = DOMAINUPPERBOUNDFACTOR * 2.0f0 * pi
    const INITIALPOSITION      = Float32[0.]
    const INITIALVELOCITY      = Float32[1.]

    main((
        NODEVECTOR = NODEVECTOR, 
        COARSEINTEGRATOR = COARSEINTEGRATOR, 
        FINEINTEGRATOR = FINEINTEGRATOR, 
        WAVENUMBER = WAVENUMBER, 
        DOMAINLOWERBOUND = DOMAINLOWERBOUND, 
        DOMAINUPPERBOUND = DOMAINUPPERBOUND,
        INITIALPOSITION = INITIALPOSITION,
        INITIALVELOCITY = INITIALVELOCITY
    ))
# end
