"""
    Propagator(propagator :: Function, discretization :: Int)

Structured representation of a propagation scheme with a discretization.
"""
struct Propagator
    propagator :: Function
    discretization :: Int
end

"""
    propagate(ivp :: FirstOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Solution

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(ivp :: FirstOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Solution
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    solution          = similar(discretizedDomain, typeof(ivp.initialValue))

    solution[1]       = ivp.initialValue
    for i in Iterators.drop(eachindex(solution), 1)
        # broadcast propagator and derivative to allow for vector values
        solution[i] = propagator.propagator.(solution[i - 1], ivp.der.(discretizedDomain[i - 1], solution[i - 1]), step) + correctors[i]
    end
    return Solution(discretizedDomain, solution)
end

"""
    propagate(ivp :: SecondOrderIVP, propagator :: Propagator) :: Solution

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(ivp :: SecondOrderIVP, propagator :: Propagator) :: Solution
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    position          = similar(discretizedDomain, ivp.initialPosition |> typeof)
    velocity          = similar(discretizedDomain, ivp.initialVelocity |> typeof)

    position[1]       = ivp.initialPosition
    velocity[1]       = ivp.initialVelocity
    for i in Iterators.drop(eachindex(position), 1)
        position[i], velocity[i] = propagator.propagator(position[i - 1], velocity[i - 1], ivp.acceleration, step)
    end
    return Solution(discretizedDomain, position, velocity)
end

"""
    propagate(ivp :: SecondOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Solution

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(
    ivp :: SecondOrderIVP, 
    propagator :: Propagator, 
    positionCorrectorVector :: Vector{Vector{Float64}}, 
    velocityCorrectorVector :: Vector{Vector{Float64}}
    ) :: Tuple{Solution, Solution}
    step                = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain   = discretize(ivp.domain, propagator.discretization)
    positionSequence    = similar(discretizedDomain, ivp.initialPosition |> typeof)
    velocitySequence    = similar(discretizedDomain, ivp.initialVelocity |> typeof)
    positionPrediction  = similar(discretizedDomain, ivp.initialPosition |> typeof)
    velocityPrediction  = similar(discretizedDomain, ivp.initialVelocity |> typeof)

    positionSequence[1] = ivp.initialPosition
    velocitySequence[1] = ivp.initialVelocity
    for i in Iterators.drop(eachindex(positionSequence), 1)
        positionPrediction[i], velocityPrediction[i] = propagator.propagator(positionSequence[i - 1], velocitySequence[i - 1], ivp.acceleration, step)
        positionSequence[i] = positionPrediction[i] + positionCorrectorVector[i - 1]
        velocitySequence[i] = velocityPrediction[i] + velocityCorrectorVector[i - 1]
    end
    pred_sol = Solution(discretizedDomain, positionPrediction, velocityPrediction)
    sol      = Solution(discretizedDomain, positionSequence, velocitySequence)
    return sol, pred_sol
end
