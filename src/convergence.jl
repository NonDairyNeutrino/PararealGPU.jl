# define the distance between solution vectors
# could either use function distance
# using LinearAlgebra: norm

function hasConverged(old_sol :: Solution, new_sol :: Solution, seq :: Symbol; threshold = 10^-10) :: Tuple{Bool, Float64}
    # the initial condition never changes so just skip it
    old_seq             = getproperty(old_sol, seq)[2:end]
    new_seq             = getproperty(new_sol, seq)[2:end]
    percentChanges      = ((n, o) -> (n ./ o) .- 1.0).(new_seq, old_seq)
    maxAbsPercentChange = percentChanges |> stack .|> abs |> maximum
    has_converged       = maxAbsPercentChange <= threshold
    return has_converged, maxAbsPercentChange
end

function hasConverged(oldSolution :: Solution, newSolution :: Solution; threshold = 10^(-10)) :: Tuple{Bool, Float64, Float64}
    position_has_converged, pos_change = hasConverged(oldSolution, newSolution, :positionSequence)
    velocity_has_converged, vel_change = hasConverged(oldSolution, newSolution, :velocitySequence)
    has_converged = position_has_converged && velocity_has_converged
    return has_converged, pos_change, vel_change
end