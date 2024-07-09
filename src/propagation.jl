struct Propagator
    propagator :: Function
    discretization :: Int
end

"""
    propagate(ivp :: InitialValueProblem, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization)) :: Vector{Vector{Float64}}

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(ivp :: InitialValueProblem, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Vector{Vector{Float64}}
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    solution          = similar(discretizedDomain)
    solution[1]       = ivp.initialValue
    for i in Iterators.drop(eachindex(solution), 1)
        solution[i] = propagator.propagator(solution[i - 1], ivp.der(discretizedDomain[i - 1], solution[i - 1]), step) + correctors[i]
    end
    return [discretizedDomain |> collect, solution]
end
