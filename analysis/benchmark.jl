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
using BenchmarkTools, DelimitedFiles
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

function bench_distributed()
    # definition of solve()
    prepCluster(NODEVECTOR, addlocal = true)

    coarse = Propagator(COARSEINTEGRATOR, COARSEDISCRETIZATION)
    fine   = Propagator(FINEINTEGRATOR,  FINEDISCRETIZATION)

    ivp = build_ivp(
        ((r, v) -> -WAVENUMBER^2 * r), 
        DOMAINLOWERBOUND, DOMAINUPPERBOUND, 
        INITIALPOSITION, INITIALVELOCITY
    )

    @info "Beginning parareal evaluation"
    sol = nothing
    iterations = 0
    try
        display(@benchmark @time "Parareal evaluation took " sol, iterations = parareal(
            ivp, 
            coarse, 
            fine; 
            threshold = sqrt(eps(Float32)), 
            localonly = false
        ))
    finally
        @info "Closing cluster."
        rmprocs(workers())
    end
    return sol, iterations
end

function main() :: Nothing
    coarse_vector      = collect(3:14)
    fine_vector        = collect(9:14)
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
    gpu_time_matrix = similar(coarse_fine_matrix, Float64)
    for index in eachindex(coarse_fine_matrix)
        coarse, fine = coarse_fine_matrix[index]
        println("Beginning benchmark for gpu with coarse = $coarse and fine = $fine")
        try
            gpu_bench = bench_gpu(2^coarse, 2^fine)
            gpu_time_matrix[index] = gpu_bench.time
        catch e
            println("Caught error for coarse = $coarse fine =$fine.")
            println("Writing -1.0 to time file.")
            println("Moving on to next discretization pair.")
            display(e)
            gpu_time_matrix[index] = -1.0
        finally
            writedlm("gpu_time_matrix.tsv", gpu_time_matrix)
        end
    end

        # printstyled("BENCHING SINGLE GPU\n", color = :green)
        # gpu_bench = bench_gpu(2^coarse, 2^fine)
        # time_tensor[2, coarse - 2, fine - 2] = gpu_bench.time

        # distributed
        # printstyled("BENCHING DISTRIBUTED\n", color = :green)
        # bench_distributed()

        # println("Single CPU: ", Base.rest(single_cpu_bench, 2))
        # println("Single GPU: ", Base.rest(gpu_bench, 2))
        # writedlm("time_matrix.tsv", time_tensor)
    return nothing
end

main()
