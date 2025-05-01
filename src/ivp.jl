"""
    Interval(lb, ub)

An object with lower and upper bounds.
"""
struct Interval
    lb :: Float64
    ub :: Float64
end

"""
    InitialValueProblem

An abstract super-type representing a first- or second-order initial value problem.
"""
abstract type InitialValueProblem end

"""
    InitialValueProblem(der, initialValue, domain)

An object representing an initial value problem
"""
struct FirstOrderIVP <: InitialValueProblem
    der          :: Function
    initialValue :: Number
    domain       :: Interval
end
IVP1 = FirstOrderIVP # type alias

"""
    SecondOrderIVP(acceleration, initialPosition, initialVelocity, domain)

An object representing a second-order initial value problem.
"""
struct SecondOrderIVP <: InitialValueProblem
    id              :: String
    domain          :: Interval
    acceleration    :: Function
    initialPosition :: Vector{Float64}
    initialVelocity :: Vector{Float64}
end
IVP2 = SecondOrderIVP # type alias

"""
    Solution(domain :: Vector{T}, positionSequence :: Vector{Vector{T}}, velocitySequence :: Vector{Vector{T}})

Structured representation of the solution to a numerical differential equation.
"""
struct Solution{T <: Real}
    domain           :: Vector{T}         # time vector
    positionSequence :: Vector{Vector{T}} # time vector of space vectors
    velocitySequence :: Vector{Vector{T}} # time vector of space vectors
    function Solution{T}(
        domain, 
        positionSequence, 
        velocitySequence = zeros(length(positionSequence) - 1)
    ) where {T <: Real}
        return new(domain, positionSequence, velocitySequence)
    end
end
function Solution(d :: Vector{T}, ps :: Vector{Vector{T}}, vs :: Vector{Vector{T}}) where {T <: Real}
    return Solution{T}(d, ps, vs)
end