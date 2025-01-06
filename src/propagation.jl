"""
    Propagator(propagator :: Function, discretization :: Int)

Structured representation of a propagation scheme with a discretization.
"""
struct Propagator
    propagator :: Function
    discretization :: Int
end
Adapt.@adapt_structure Propagator
# """
#     Adapt.adapt_structure(to, itp::Propagator)

# TBW
# """
# function Adapt.adapt_structure(to, prop::Propagator)
#     propagator = Adapt.adapt_structure(to, prop.propagator)
#     discretization = Adapt.adapt_structure(to, prop.discretization)
#     Propagator(propagator, discretization)
# end

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
Adapt.@adapt_structure Solution
# """
#     Adapt.adapt_structure(to, itp::Solution)

# TBW
# """
# function Adapt.adapt_structure(to, sol::Solution)
#     domain = Adapt.adapt_structure(to, sol.domain)
#     range = Adapt.adapt_structure(to, sol.range)
#     Solution(domain, range)
# end

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
function propagate(ivp :: SecondOrderIVP, propagator :: Propagator, rangeCorrectors :: Vector{Float64} = zeros(propagator.discretization + 1), derivativeCorrectors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Solution
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    range             = similar(discretizedDomain, ivp.initialPosition |> typeof)
    derivative        = similar(discretizedDomain, ivp.initialVelocity |> typeof)

    range[1]          = ivp.initialPosition
    derivative[1]     = ivp.initialVelocity
    for i in Iterators.drop(eachindex(range), 1)
        range[i], derivative[i] = propagator.propagator(range[i - 1], derivative[i - 1], ivp.acceleration, step)
        range[i]      += rangeCorrectors[i]
        derivative[i] += derivativeCorrectors[i]
    end
    return Solution(discretizedDomain, range, derivative)
end
