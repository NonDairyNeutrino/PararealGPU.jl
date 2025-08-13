module PararealGPUBenchmarks
export bench_single, bench_all_single, bench_gpu, bench_all_gpu, bench_distributed, bench_all_distributed

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

function bench_gpu(coarse :: Int, fine :: Int) :: NamedTuple
    bench = @btimed solve(
            NODEVECTOR,
            COARSEINTEGRATOR,
            $coarse,
            FINEINTEGRATOR,
            $fine,
            ((r, v) -> -WAVENUMBER^2 * r),
            DOMAINLOWERBOUND,
            DOMAINUPPERBOUND,
            INITIALPOSITION,
            INITIALVELOCITY;
            addlocal  = true,
            localonly = true
        )
    return bench
end

function bench_all_gpu(coarse_fine_matrix :: Matrix{Tuple{Int, Int}}; time_file_name :: String = "") :: Tuple{Matrix{Float64}}
    time_matrix = similar(coarse_fine_matrix, Float64)

    for index in eachindex(coarse_fine_matrix)
        coarse, fine = coarse_fine_matrix[index]
        println("Beginning benchmark for gpu with coarse = $coarse and fine = $fine")

        sol = (Solution(Float64[], [Float64[]], [Float64[]]), 0)
        try
            bench = bench_gpu(2^coarse, 2^fine)
            time_matrix[index] = bench.time
            sol                = bench.value

        catch e
            println("Caught error for coarse = $coarse fine =$fine.")
            println("Writing -1.0 to time file.")
            println("Moving on to next discretization pair.")
            display(e)
            time_matrix[index] = -1.0
            # if error write outer sol to file

        finally

            if !isempty(time_file_name)
                time_file = DATADIR * time_file_name
                writedlm(time_file, time_matrix)
                println("GPU runtimes saved to ", time_file)
            end

            sol_file = DATADIR * "sol_gpu_c$(coarse)_f$(fine).jld2"
            save_object(sol_file, sol)
            println("GPU solutions saved to ", sol_file)
        end
    end

    return time_matrix
end

function bench_distributed(coarse_disc :: Int, fine_disc :: Int) :: NamedTuple
    coarse = PararealGPU.Propagator(COARSEINTEGRATOR, coarse_disc)
    fine   = PararealGPU.Propagator(FINEINTEGRATOR,  fine_disc)
    bench  = @btimed PararealGPU.parareal(
        $ivp, 
        $coarse, 
        $fine
    )
    return bench
end

function bench_all_distributed(coarse_fine_matrix :: Matrix{Tuple{Int, Int}}; time_file_name :: String = "") :: Tuple{Matrix{Float64}}
    time_matrix = similar(coarse_fine_matrix, Float64)

    PararealGPU.prepCluster(NODEVECTOR, addlocal = true)
    ivp = PararealGPU.build_ivp(
        ((r, v) -> -WAVENUMBER^2 * r), 
        DOMAINLOWERBOUND, DOMAINUPPERBOUND, 
        INITIALPOSITION, INITIALVELOCITY
    )

    for index in eachindex(coarse_fine_matrix)
        coarse, fine = coarse_fine_matrix[index]
        println("Beginning benchmark for gpu with coarse = $coarse and fine = $fine")

        sol = (Solution(Float64[], [Float64[]], [Float64[]]), 0)
        try
            bench = bench_distributed(2^coarse, 2^fine)
            time_matrix[index] = bench.time
            sol                = bench.value

        catch e
            println("Caught error for coarse = $coarse fine =$fine.")
            println("Writing -1.0 to time file.")
            println("Moving on to next discretization pair.")
            display(e)
            time_matrix[index] = -1.0
            # if error write outer sol to file
        finally
            if !isempty(time_file_name)
                time_file = DATADIR * time_file_name
                writedlm(time_file, time_matrix)
                println("Distributed runtimes saved to ", time_file)
            end

            sol_file = DATADIR * "sol_dist_c$(coarse)_f$(fine).jld2"
            save_object(sol_file, sol)
            println("Distributed solutions saved to ", sol_file)
        end
    end

    return time_matrix
end

function main() :: Nothing
    coarse_vector      = 3:14
    fine_vector        = 3:14
    coarse_fine_matrix = Iterators.product(coarse_vector, fine_vector) |> collect

    save_object(DATADIR * "disc_matrix.jld2", coarse_fine_matrix)
    bench_all_single(coarse_vector)
    # bench_all_gpu(coarse_fine_matrix;         time_file_name = "gpu_time_matrix.tsv")
    # bench_all_distributed(coarse_fine_matrix; time_file_name = "dist_time_matrix.tsv")

    return nothing
end

# if this file is explicitly run, then actually do the benchmarks
if abspath(PROGRAM_FILE) == @__FILE__
    # DEFINE CLUSTER
    const NODEVECTOR           = String["Electromagnetism"]
    # DEFINE COMPUTATIONAL PARAMETERS
    const COARSEINTEGRATOR     = symplecticEuler
    # const COARSEDISCRETIZATION = 2^10                        # how many total problems
    const FINEINTEGRATOR       = velocityVerlet
    # const FINEDISCRETIZATION   = 2^10                         # 2^10 = 1024 steps -> each step is ~0.01% of the domain
    # DEFINE MODEL PARAMETERS
    const WAVENUMBER           = 1.0f0 # * pi # DO NO CHANGE
    # ACCELERATION(r, v)         = -WAVENUMBER^2 * r         # simple harmonic oscillator
    const DOMAINLOWERBOUND     = 0.0f0
    const DOMAINUPPERBOUNDFACTOR = 10
    const DOMAINUPPERBOUND     = DOMAINUPPERBOUNDFACTOR * 2.0f0 * pi
    const INITIALPOSITION      = Float32[0.]
    const INITIALVELOCITY      = Float32[1.]

    main()
end

end
