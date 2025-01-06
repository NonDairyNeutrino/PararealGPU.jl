"""
    Interval(lb, ub)

An object with lower and upper bounds.
"""
struct Interval
    lb :: Float64
    ub :: Float64
end
# """
#     Adapt.adapt_structure(to, itp::Interval)

# TBW
# """
# function Adapt.adapt_structure(to, interval::Interval)
#     lb = Adapt.adapt_structure(to, interval.lb)
#     ub = Adapt.adapt_structure(to, interval.ub)
#     Interval(lb, ub)
# end

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
# """
#     Adapt.adapt_structure(to, ivp::FirstOrderIVP)

# TBW
# """
# function Adapt.adapt_structure(to, ivp::FirstOrderIVP)
#     der          = Adapt.adapt_structure(to, ivp.der)
#     initialValue = Adapt.adapt_structure(to, ivp.initialValue)
#     domain       = Adapt.adapt_structure(to, ivp.domain)
#     FirstOrderIVP(der, initialValue, domain)
# end

"""
    SecondOrderIVP(acceleration, initialPosition, initialVelocity, domain)

An object representing a second-order initial value problem.
"""
struct SecondOrderIVP <: InitialValueProblem
    domain          :: Interval
    acceleration    :: Function
    initialPosition :: Vector{Float64}
    initialVelocity :: Vector{Float64}
end
IVP2 = SecondOrderIVP # type alias
# """
#     Adapt.adapt_structure(to, ivp::SecondOrderIVP)

# TBW
# """
# function Adapt.adapt_structure(to, ivp::SecondOrderIVP)
#     acceleration    = Adapt.adapt_structure(to, ivp.acceleration)
#     initialPosition = Adapt.adapt_structure(to, ivp.initialPosition)
#     initialVelocity = Adapt.adapt_structure(to, ivp.initialVelocity)
#     domain          = Adapt.adapt_structure(to, ivp.domain)
#     SecondOrderIVP(acceleration, initialPosition, initialVelocity, domain)
# end
