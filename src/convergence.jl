"""
    hasConverged(
        old_sol :: Solution{T}, 
        new_sol :: Solution{T}, 
        seq :: Symbol; 
        threshold :: T = convert(T, 1.0e-10)
    ) :: Tuple{Bool, T} where T <: AbstractFloat

Determines convergence of the given property given the previous and and current solutions.
"""
function hasConverged(
        old_sol :: Solution{T}, 
        new_sol :: Solution{T}, 
        seq :: Symbol; 
        threshold :: T = convert(T, 1.0e-10)
    ) :: Tuple{Bool, T} where T <: AbstractFloat
    # the initial condition never changes so just skip it
    old_seq             = getproperty(old_sol, seq)[2:end]
    new_seq             = getproperty(new_sol, seq)[2:end]
    percentChanges      = ((n, o) -> (n ./ o) .- 1.0).(new_seq, old_seq)
    maxAbsPercentChange = percentChanges |> stack .|> abs |> maximum
    has_converged       = maxAbsPercentChange <= threshold
    return has_converged, maxAbsPercentChange
end

"""
    hasConverged(
        oldSolution :: Solution{T}, 
        newSolution :: Solution{T}; 
        threshold = convert(T, 1.0e-10)
    ) :: Tuple{Bool, T, T} where T <: AbstractFloat

Determines convergence of both position and velocity given the previous and current solutions.
"""
function hasConverged(
        oldSolution :: Solution{T}, 
        newSolution :: Solution{T}; 
        threshold = convert(T, 1.0e-10)
    ) :: Tuple{Bool, T, T} where T <: AbstractFloat
    position_has_converged, pos_change = hasConverged(oldSolution, newSolution, :positionSequence)
    velocity_has_converged, vel_change = hasConverged(oldSolution, newSolution, :velocitySequence)
    has_converged                      = position_has_converged && velocity_has_converged
    return has_converged, pos_change, vel_change
end
