"""
    build_ivp(
        acc                  :: Function,
        lowerBound           :: T,
        upperBound           :: T,
        initialPosition      :: Vector{T},
        initialVelocity      :: Vector{T}
    ) :: SecondOrderIVP{T} where T <: AbstractFloat

Build an initial value problem with ID 0.
"""
function build_ivp(
        acc                  :: Function,
        lowerBound           :: T,
        upperBound           :: T,
        initialPosition      :: Vector{T},
        initialVelocity      :: Vector{T}
    ) :: SecondOrderIVP{T} where T <: AbstractFloat
    @info "Creating initial value problem"

    function acceleration(
            position :: Vector{T}, velocity :: Vector{T}
        ) :: Vector{T} where T <: AbstractFloat
        return acc(position, velocity)
    end
    domain = Interval{T}(lowerBound, upperBound)
    ivp    = SecondOrderIVP("0", domain, acceleration, initialPosition, initialVelocity)
    return ivp
end

"""
    solve(
        nodeVector           :: Vector{String},
        coarseIntegrator     :: Function,
        coarseDiscretization :: Int,
        fineIntegrator       :: Function,
        fineDiscretization   :: Int,
        acc                  :: Function,
        lowerBound           :: T,
        upperBound           :: T,
        initialPosition      :: Vector{T},
        initialVelocity      :: Vector{T};
        addlocal             :: Bool = false,
        threshold            :: T    = convert(T, 10)
    ) :: Solution{T} where T <: AbstractFloat

Solves the given problem using the given coarse and fine propagators.
"""
function solve(
        nodeVector           :: Vector{String},
        coarseIntegrator     :: Function,
        coarseDiscretization :: Int,
        fineIntegrator       :: Function,
        fineDiscretization   :: Int,
        acc                  :: Function,
        lowerBound           :: T,
        upperBound           :: T,
        initialPosition      :: Vector{T},
        initialVelocity      :: Vector{T};
        addlocal             :: Bool = false,
        localonly            :: Bool = false,
        threshold            :: T    = max(eps(T), 10^-10),
        initialSolution      :: String = ""
    ) :: Tuple{Solution{T}, Int} where T <: AbstractFloat
    !isempty(initialSolution) && @warn "Starting from checkpointed solution: $initialSolution"
    sizeof(T) > 4 && @warn "Floats are larger than 32 bits. Consider downsizing to increase GPU performance." T

    !localonly && prepCluster(nodeVector, addlocal = addlocal)

    coarse = Propagator(coarseIntegrator, coarseDiscretization)
    fine   = Propagator(fineIntegrator,  fineDiscretization)

    ivp = build_ivp(
        acc, lowerBound, upperBound, initialPosition, initialVelocity
    )

    @info "Beginning parareal evaluation"
    sol = nothing
    iterations = 0
    try
        @time "Parareal evaluation took " sol, iterations = parareal(
            ivp, 
            coarse, 
            fine; 
            threshold = threshold, 
            localonly = localonly, 
            initialSolution = initialSolution
        )
    finally
        @info "Closing cluster."
        !localonly && rmprocs(workers())
    end
    return sol, iterations
end
