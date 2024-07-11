struct Propagator
    propagator :: Function
    discretization :: Int
end

"""
    propagate(ivp :: InitialValueProblem, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization)) :: Vector{Vector{Float64}}

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(ivp :: FirstOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Vector{Vector{Float64}}
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    solution          = similar(discretizedDomain)
    solution[1]       = ivp.initialValue
    for i in Iterators.drop(eachindex(solution), 1)
        solution[i] = propagator.propagator(solution[i - 1], ivp.der(discretizedDomain[i - 1], solution[i - 1]), step) + correctors[i]
    end
    return [discretizedDomain |> collect, solution]
end

"""
    propagate(ivp :: SecondOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Vector{Vector{Float64}}

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(ivp :: SecondOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Vector{Vector{Float64}}
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    solution          = similar(discretizedDomain)

    solution[1]       = ivp.initialPosition
    solution[2]       = ivp.initialPosition + ivp.initialVelocity * step + 0.5 * ivp.acceleration(ivp.initialPosition, ivp.initialVelocity) * step^2 # verlet needs two points
    for i in Iterators.drop(eachindex(solution), 2)
        # This implementation currently assumes the propagator has the same signature as the "verlet" propagator
        velocity    = (solution[i - 1] - solution[i - 2]) / step # FIXME: find a better method to get the velocity or find a workaround
        solution[i] = propagator.propagator(solution[i - 1], solution[i - 2], ivp.acceleration(solution[i], velocity), step) + correctors[i]
    end
    return [discretizedDomain |> collect, solution]
end
