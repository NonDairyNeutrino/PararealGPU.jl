# define the distance between solution vectors
# could either use function distance
# using LinearAlgebra: norm

function hasConverged(oldSolution :: Solution, newSolution :: Solution; threshold = 10^(-10)) :: Bool
    # position convergence test
    oldPositionSequence = oldSolution.positionSequence
    newPositionSequence = newSolution.positionSequence

    position_has_converged = all(v -> maximum(v) <= threshold, newPositionSequence - oldPositionSequence)
    # any(norm.(newPositionSequence - oldPositionSequence) .>= threshold) && return false

    # velocity convergence test
    oldVelocitySequence = oldSolution.velocitySequence
    newVelocitySequence = newSolution.velocitySequence
    velocity_has_converged = all(v -> maximum(v) <= threshold, newVelocitySequence - oldVelocitySequence)
    # any(norm.(newVelocitySequence - oldVelocitySequence) .>= threshold) && return false
    has_converged = position_has_converged && velocity_has_converged
    # has_converged && printstyled("Parareal has converged.\n", color=:green)
    return has_converged
end