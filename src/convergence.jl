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
    old_seq             = getproperty(old_sol, seq)[2:end] |> stack
    new_seq             = getproperty(new_sol, seq)[2:end] |> stack
    @assert all(!iszero, old_seq) "Old sequence contains a zero at $(findall(iszero, old_seq)). This is will cause a NaN."
    maxAbsPercentChange = maximum(@. abs(new_seq / old_seq - convert(T, 1)))
    @assert !isnan(maxAbsPercentChange) "Somehow maxAbsPercentChange is NaN.  That is not good."
    iszero(maxAbsPercentChange) && @warn "Solution did not change.  Be weary of these results."
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
