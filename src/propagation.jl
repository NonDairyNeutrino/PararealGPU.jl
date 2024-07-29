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
    domain :: Vector{Float64}
    range  :: Union{Vector{Float64}, Vector{Vector{Float64}}}
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
function propagate(ivp :: SecondOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Solution
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    solution          = similar(discretizedDomain, ivp.initialPosition)
    derivative        = similar(discretizedDomain, ivp.initialVelocity)

    solution[1]       = ivp.initialPosition
    derivative[1]     = ivp.initialVelocity
    for i in Iterators.drop(eachindex(solution), 2)
        solution[i], derivative[i] = propagator.propagator(solution[i - 1], derivative[i - 1], ivp.acceleration(solution[i - 1], derivative[i - 1])) + correctors[i]
    end
    return Solution(discretizedDomain, solution)
end
