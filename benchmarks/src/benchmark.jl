module PararealGPUBenchmarks
export bench_single, bench_all_single, bench_gpu, bench_all_gpu, bench_distributed, bench_all_distributed

using BenchmarkTools, DelimitedFiles, Distributed, JLD2
const proj_dir = Base.active_project() |> dirname
include("$proj_dir/src/PararealGPU.jl"); using .PararealGPU

const DATADIR = dirname(@__DIR__) * "/data/"

"""
    bench_single(coarse :: Int, fine :: Int) :: NamedTuple

TBW
"""
function bench_single(coarse :: Int, fine :: Int) :: NamedTuple
    maxsteps = coarse * fine
    time_step = (DOMAINUPPERBOUND - DOMAINLOWERBOUND) / maxsteps
    bench = @btimed begin
        for i in 1:$maxsteps
            pos, vel = FINEINTEGRATOR(pos, vel, (x, v) -> -WAVENUMBER * x, $time_step)
        end
    end setup=(pos = INITIALPOSITION; vel = INITIALVELOCITY;)
    return bench
end

"""
    bench_all_single(coarse_fine_matrix :: Matrix{Int}; file_name :: String = "") :: Matrix{Float64}

Benchmark all single threaded cases in parallel and return their results, optionally writing the results to a file.
"""
function bench_all_single(coarse_fine_matrix :: Matrix{Int}; time_file_name :: String = "", sol_file_name :: String = "") :: Tuple{
        Matrix{Float64},
        Matrix{Tuple{Solution, Int}}
    }

    time_matrix = similar(coarse_fine_matrix, Float64)
    sol_matrix  = similar(coarse_fine_matrix, Tuple{Solution, Int})

    Threads.@threads for index in eachindex(coarse_fine_matrix)
        coarse, fine = coarse_fine_matrix[index]
        println("Beginning benchmark for single threaded with coarse = $coarse and fine = $fine")

        try
            bench = bench_single(2^coarse, 2^fine)
            time_matrix[index] = bench.time
            sol_matrix[index]  = bench.value
        catch e
            println("Caught error for coarse = $coarse fine = $fine.")
            println("Writing -1.0 to time file.")
            println("Moving on to next discretization pair.")
            display(e)
            time_matrix[index] = -1.0
            sol_matrix[index]  = (Solution(Float64[], [Float64[]], [Float64[]]), 0)
        finally
            if !isempty(time_file_name)
                time_file = DATADIR * time_file_name
                writedlm(time_file, time_matrix)
                println("Single threaded runtimes saved to ", time_file)
            end

            if !isempty(sol_file_name)
                sol_file_name *= sol_file_name[end-3 : end] == "jld2" ? "" : "jld2"
                sol_file = DATADIR * sol_file_name
                jldsave(sol_file, sol_matrix)
                println("Single threaded solutions saved to ", sol_file)
            end
        end

    end

    return time_matrix, sol_matrix
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

function bench_all_gpu(coarse_fine_matrix :: Matrix{Int}; time_file_name :: String = "", sol_file_name :: String = "") :: Tuple{
        Matrix{Float64},
        Matrix{Tuple{Solution, Int}}
    }
    time_matrix = similar(coarse_fine_matrix, Float64)
    sol_matrix  = similar(coarse_fine_matrix, Tuple{Solution, Int})

    for index in eachindex(coarse_fine_matrix)
        coarse, fine = coarse_fine_matrix[index]
        println("Beginning benchmark for gpu with coarse = $coarse and fine = $fine")
        try
            bench = bench_gpu(2^coarse, 2^fine)
            time_matrix[index] = bench.time
            sol_matrix[index]  = bench.value
        catch e
            println("Caught error for coarse = $coarse fine =$fine.")
            println("Writing -1.0 to time file.")
            println("Moving on to next discretization pair.")
            display(e)
            time_matrix[index] = -1.0
            sol_matrix[index]  = (Solution(Float64[], [Float64[]], [Float64[]]), 0)
        finally
            if !isempty(time_file_name)
                time_file = DATADIR * time_file_name
                writedlm(time_file, time_matrix)
                println("GPU runtimes saved to ", time_file)
            end

            if !isempty(sol_file_name)
                sol_file_name *= sol_file_name[end-3 : end] == "jld2" ? "" : "jld2"
                sol_file = DATADIR * sol_file_name
                jldsave(sol_file, sol_matrix)
                println("GPU solutions saved to ", sol_file)
            end
        end
    end

    return time_matrix, sol_matrix
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

function bench_all_distributed(coarse_fine_matrix :: Matrix{Int}; time_file_name :: String = "", sol_file_name :: String = "") :: Tuple{
        Matrix{Float64},
        Matrix{Tuple{Solution, Int}}
    }
    time_matrix = similar(coarse_fine_matrix, Float64)
    sol_matrix  = similar(coarse_fine_matrix, Tuple{Solution, Int})

    PararealGPU.prepCluster(NODEVECTOR, addlocal = true)
    ivp = PararealGPU.build_ivp(
        ((r, v) -> -WAVENUMBER^2 * r), 
        DOMAINLOWERBOUND, DOMAINUPPERBOUND, 
        INITIALPOSITION, INITIALVELOCITY
    )

    for index in eachindex(coarse_fine_matrix)
        coarse, fine = coarse_fine_matrix[index]
        println("Beginning benchmark for gpu with coarse = $coarse and fine = $fine")
        try
            bench = bench_distributed(2^coarse, 2^fine)
            time_matrix[index] = bench.time
            sol_matrix[index]  = bench.value
        catch e
            println("Caught error for coarse = $coarse fine =$fine.")
            println("Writing -1.0 to time file.")
            println("Moving on to next discretization pair.")
            display(e)
            time_matrix[index] = -1.0
            sol_matrix[index]  = (Solution(Float64[], [Float64[]], [Float64[]]), 0)
        finally
            if !isempty(time_file_name)
                time_file = DATADIR * time_file_name
                writedlm(time_file, time_matrix)
                println("Distributed runtimes saved to ", time_file)
            end

            if !isempty(sol_file_name)
                sol_file_name *= sol_file_name[end-3 : end] == "jld2" ? "" : "jld2"
                sol_file = DATADIR * sol_file_name
                jldsave(sol_file, sol_matrix)
                println("Single threaded solutions saved to ", sol_file)
            end
        end
    end

    return time_matrix, sol_matrix
end

function main() :: Nothing
    coarse_vector      = 3:14
    fine_vector        = 3:14
    coarse_fine_matrix = Iterators.product(coarse_vector, fine_vector) |> collect

    writedlm(DATADIR * "disc_matrix.tsv", coarse_fine_matrix)
    bench_all_single(coarse_fine_matrix;      time_file_name = "single_time_matrix.tsv", sol_file_name = "single_sol_matrix.jld2")
    bench_all_gpu(coarse_fine_matrix;         time_file_name = "gpu_time_matrix.tsv",    sol_file_name = "gpu_sol_matrix.jld2")
    bench_all_distributed(coarse_fine_matrix; time_file_name = "dist_time_matrix.tsv",   sol_file_name = "dist_sol_matrix.jld2")

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