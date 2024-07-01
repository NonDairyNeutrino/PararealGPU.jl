"""
    Interval(lb, ub)

An object with lower and upper bounds.
"""
struct Interval
    lb :: Float64
    ub :: Float64
end

"""
    InitialValueProblem(der, initialValue, domain)

An object representing an initial value problem
"""
struct InitialValueProblem
    der :: Function
    initialValue :: Number
    domain :: Interval
end