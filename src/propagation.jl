"""
    coarsePropagate(propagate :: Function, domain :: Interval, initialValue :: Float64) :: Vector{Vector{Float64}}

Coarsely propagate a value on an interval.
"""
function coarsePropagate(propagate :: Function, domain :: Interval, initialValue :: Float64) :: Vector{Vector{Float64}}
    step              = (domain.ub - domain.lb) / COARSEDISCRETIZATION
    discretizedDomain = discretize(domain, COARSEDISCRETIZATION)
    solution          = similar(discretizedDomain)
    solution[1] = initialValue
    for i in 2:COARSEDISCRETIZATION + 1
        solution[i] = propagate(solution[i - 1], der(discretizedDomain[i - 1], solution[i - 1]), step)
    end
    return [discretizedDomain |> collect, solution]
end

"""
    finePropagate(propagate :: Function, domain :: Interval, initialValue :: Float64) :: Vector{Vector{Float64}}

Finely propagate a value on an interval.
"""
function finePropagate(propagate :: Function, domain :: Interval, initialValue :: Float64) :: Vector{Vector{Float64}}
    step              = (domain.ub - domain.lb) / FINEDISCRETIZATION
    discretizedDomain = discretize(domain, FINEDISCRETIZATION)
    solution          = similar(discretizedDomain)
    solution[1] = initialValue
    for i in 2:FINEDISCRETIZATION + 1
        solution[i] = propagate(solution[i - 1], der(discretizedDomain[i - 1], solution[i - 1]), step)
    end
    return [discretizedDomain |> collect, solution]
end

"""
    correct(domain :: Interval, initialValue :: Float64)

Coarsely propagate the initial value with corrections.
"""
function correct(propagate :: Function, correctors ::Vector{Float64}) :: Vector{Float64}
    step              = (DOMAIN.ub - DOMAIN.lb) / COARSEDISCRETIZATION
    discretizedDomain = discretize(DOMAIN, COARSEDISCRETIZATION)
    solution          = similar(discretizedDomain)
    solution[1] = INITIALVALUE
    for i in 2:COARSEDISCRETIZATION + 1
        solution[i] = propagate(solution[i - 1], der(discretizedDomain[i - 1], solution[i - 1]), step) + correctors[i]
    end
    return solution
end
