"""
    Propagator(propagator :: Function, discretization :: Int)

Structured representation of a propagation scheme with a discretization.
"""
struct Propagator
    propagator :: Function
    discretization :: Int
end

"""
    Solution(domain :: Vector{Float64}, range :: Any)

Structured representation of the solution to a numerical differential equation.
"""
struct Solution
    domain   :: Vector{Float64}         # time vector
    position :: Vector{Vector{Float64}} # time vector of space vectors
    velocity :: Vector{Vector{Float64}} # time vector of space vectors
    function Solution(domain, range :: Vector{Vector{Float64}}, derivative = zeros(length(range) - 1) :: Vector{Vector{Float64}})
        return new(domain, range, derivative)
    end
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
    propagate(ivp :: SecondOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Solution

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(ivp :: SecondOrderIVP, propagator :: Propagator, rangeCorrectors :: Vector{Vector{Float64}} = fill(zeros(ivp.initialPosition |> length), propagator.discretization + 1), derivativeCorrectors :: Vector{Vector{Float64}} = fill(zeros(ivp.initialVelocity |> length), propagator.discretization + 1)) :: Solution
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    position          = similar(discretizedDomain, ivp.initialPosition |> typeof)
    velocity          = similar(discretizedDomain, ivp.initialVelocity |> typeof)

    position[1]       = ivp.initialPosition
    velocity[1]       = ivp.initialVelocity
    for i in Iterators.drop(eachindex(position), 1)
        position[i], velocity[i] = propagator.propagator(position[i - 1], velocity[i - 1], ivp.acceleration, step)
        position[i] += rangeCorrectors[i]
        velocity[i] += derivativeCorrectors[i]
    end
    return Solution(discretizedDomain, position, velocity)
end
