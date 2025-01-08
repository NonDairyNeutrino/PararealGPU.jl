# define the distance between solution vectors
# could either use function distance
using LinearAlgebra: norm

function hasConverged(oldSolution :: Solution, newSolution :: Solution; threshold = 10^(-10)) :: Bool
    # position convergence test
    oldPositionSequence = oldSolution.positionSequence
    newPositionSequence = newSolution.positionSequence
    any(norm.(newPositionSequence - oldPositionSequence) .>= threshold) && return false

    # velocity convergence test
    oldVelocitySequence = oldSolution.velocitySequence
    newVelocitySequence = newSolution.velocitySequence
    any(norm.(newVelocitySequence - oldVelocitySequence) .>= threshold) && return false

    return true
end