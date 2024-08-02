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
    acceleration    :: Function
    initialPosition :: Float64
    initialVelocity :: Float64
    domain          :: Interval
end
IVP2 = SecondOrderIVP # type alias

