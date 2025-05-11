"""
    Interval{T <: AbstractFloat}(lb :: T, ub :: T)

An object with lower and upper bounds.
"""
struct Interval{T <: AbstractFloat}
    lb :: T
    ub :: T
end

"""
    InitialValueProblem

An abstract super-type representing a first- or second-order initial value problem.
"""
abstract type InitialValueProblem end

"""
    InitialValueProblem(der :: Function, initialValue :: Number, domain :: Interval)

An object representing an initial value problem
"""
struct FirstOrderIVP <: InitialValueProblem
    der          :: Function
    initialValue :: Number
    domain       :: Interval
end
IVP1 = FirstOrderIVP # type alias

"""
    SecondOrderIVP{T <: AbstractFloat}(acceleration, initialPosition, initialVelocity, domain)

An object representing a second-order initial value problem.
"""
struct SecondOrderIVP{T <: AbstractFloat} <: InitialValueProblem
    # TODO: make mutable to only change the fields instead of creating new ones every time
    id              :: String
    domain          :: Interval{T}
    acceleration    :: Function
    initialPosition :: Vector{T}
    initialVelocity :: Vector{T}
end
IVP2 = SecondOrderIVP # type alias

"""
    Solution(domain :: Vector{T}, positionSequence :: Vector{Vector{T}}, velocitySequence :: Vector{Vector{T}})

Structured representation of the solution to a numerical differential equation.
"""
struct Solution{T <: AbstractFloat}
    # TODO: make mutable to only change the fields instead of creating new ones every time
    domain           :: Vector{T}         # time vector
    positionSequence :: Vector{Vector{T}} # time vector of space vectors
    velocitySequence :: Vector{Vector{T}} # time vector of space vectors
    function Solution{T}(
        domain, 
        positionSequence, 
        velocitySequence = zeros(length(positionSequence) - 1)
    ) where T <: AbstractFloat
        return new(domain, positionSequence, velocitySequence)
    end
end
function Solution(d :: Vector{T}, ps :: Vector{Vector{T}}, vs :: Vector{Vector{T}}) where {T <: AbstractFloat}
    return Solution{T}(d, ps, vs)
end