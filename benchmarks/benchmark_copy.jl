#= 
Compare the timings of the implementation between how many threads are used.
Comparing between
- CPU single threaded
    - just straight velocityVerlet
- CPU multithreaded
    - CPU parareal
- GPU multithreaded
    - local GPU parareal
- distributed
    - distributed parareal
=#

using Plots: plot, plot!, savefig
using BenchmarkTools, DelimitedFiles, Distributed
include("$(pwd())/src/PararealGPU.jl"); using .PararealGPU

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

function bench_single_cpu(coarse :: Int, fine :: Int) :: NamedTuple
    maxsteps = coarse * fine
    time_step = (DOMAINUPPERBOUND - DOMAINLOWERBOUND) / maxsteps
    bench = @btimed begin
        for i in 1:$maxsteps
            pos, vel = FINEINTEGRATOR(pos, vel, (x, v) -> -WAVENUMBER * x, $time_step)
        end
    end setup=(pos = INITIALPOSITION; vel = INITIALVELOCITY;)
    return bench
end

function bench_multi_cpu()
    
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
            localonly = true,
            # eps(Float32) == 1.1920929f-7
            # sqrt(eps(Float32)) == 0.00034526698f0
            # this mirrors isapprox()
            threshold = sqrt(eps(Float32))
        )
    return bench
end

PararealGPU.prepCluster(NODEVECTOR, addlocal = true)
ivp = PararealGPU.build_ivp(
        ((r, v) -> -WAVENUMBER^2 * r), 
        DOMAINLOWERBOUND, DOMAINUPPERBOUND, 
        INITIALPOSITION, INITIALVELOCITY
    )

function bench_distributed(coarse_disc :: Int, fine_disc :: Int) :: NamedTuple
    coarse = PararealGPU.Propagator(COARSEINTEGRATOR, coarse_disc)
    fine   = PararealGPU.Propagator(FINEINTEGRATOR,  fine_disc)
    bench  = @btimed PararealGPU.parareal(
        $ivp, 
        $coarse, 
        $fine; 
        threshold = sqrt(eps(Float32)), 
        localonly = false
    )
    return bench
end

function main() :: Nothing
    coarse_vector      = collect(3:14)
    fine_vector        = collect(12:14)
    coarse_fine_matrix = Iterators.product(coarse_vector, fine_vector) |> collect

    # bench and write all single threaded benchmarks before doing parallelized methods
    # single_time_matrix = similar(coarse_fine_matrix, Float64)
    # Threads.@threads for index in eachindex(coarse_fine_matrix)
    #     coarse, fine = coarse_fine_matrix[index]
    #     println("Beginning benchmark for single threaded with coarse = $coarse and fine = $fine")
    #     single_bench = bench_single_cpu(2^coarse, 2^fine)
    #     single_time_matrix[index] = single_bench.time
    # end
    # writedlm("single_time_matrix.tsv", single_time_matrix)

        # multi-threaded cpu
        # printstyled("BENCHING MULTI-THREADED CPU", color = :green)
        # multi_cpu()

    # bench gpu for all discretizations
    # gpu_time_matrix = similar(coarse_fine_matrix, Float64)
    # for index in eachindex(coarse_fine_matrix)
    #     coarse, fine = coarse_fine_matrix[index]
    #     println("Beginning benchmark for gpu with coarse = $coarse and fine = $fine")
    #     gpu_bench = bench_gpu(2^coarse, 2^fine)
    #     single_time_matrix[index] = gpu_bench.time
    # end
    # writedlm("gpu_time_matrix.tsv", gpu_time_matrix)

    # bench distributed for all discretizations
    distributed_time_matrix = similar(coarse_fine_matrix, Float64)
    for index in eachindex(coarse_fine_matrix)
        coarse, fine = coarse_fine_matrix[index]
        println("Beginning benchmark for distributed with coarse = $coarse and fine = $fine")
        try 
            distributed_bench = bench_distributed(2^coarse, 2^fine)
            distributed_time_matrix[index] = distributed_bench.time
        catch e
            println("Caught error for coarse = $coarse fine =$fine.")
            println("Writing -1.0 to time file.")
            println("Moving on to next discretization pair.")
            display(e)
            distributed_time_matrix[index] = -1.0
        finally
            writedlm("distributed_time_matrix.tsv", distributed_time_matrix)
        end
    end

    return nothing
end

main()
