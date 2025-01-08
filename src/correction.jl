# functionality for the correction phase of the parareal algorithm
"""
    correctPosition!(
    subSolutionFineVector   :: Vector{Solution}, 
    subSolutionCoarseVector :: Vector{Solution},
    positionCorrectorVector :: Vector{Vector{Float64}}
    )

Correct solutions positions.
"""
function correctPosition!(
    subSolutionFineVector   :: Vector{Solution}, 
    subSolutionCoarseVector :: Vector{Solution},
    positionCorrectorVector :: Vector{Vector{Float64}}
    )
    for i in eachindex(positionCorrectorVector)
        fineCorrector              = subSolutionFineVector[i].positionSequence[end]
        coarseCorrector            = subSolutionCoarseVector[i].positionSequence[end]
        positionCorrectorVector[i] = fineCorrector - coarseCorrector
    end
end

"""
    correctVelocity!(
    subSolutionFineVector   :: Vector{Solution}, 
    subSolutionCoarseVector :: Vector{Solution},
    velocityCorrectorVector :: Vector{Vector{Float64}}
    )

Correct solutions velocities.
"""
function correctVelocity!(
    subSolutionFineVector   :: Vector{Solution}, 
    subSolutionCoarseVector :: Vector{Solution},
    velocityCorrectorVector :: Vector{Vector{Float64}}
    )
    for i in eachindex(velocityCorrectorVector)
        fineCorrector              = subSolutionFineVector[i].velocitySequence[end]
        coarseCorrector            = subSolutionCoarseVector[i].velocitySequence[end]
        velocityCorrectorVector[i] = fineCorrector - coarseCorrector
    end
end

"""
    correct!(
    subProblemVector        :: Vector{SecondOrderIVP}, 
    subSolutionFineVector   :: Vector{Solution}, 
    subSolutionCoarseVector :: Vector{Solution},
    positionCorrectorVector :: Vector{Vector{Float64}}
    )

Correct solutions positions and velocities.
"""
function correct!(
    subSolutionFineVector   :: Vector{Solution}, 
    subSolutionCoarseVector :: Vector{Solution},
    positionCorrectorVector :: Vector{Vector{Float64}},
    velocityCorrectorVector :: Vector{Vector{Float64}}
    )
    correctPosition!(subSolutionFineVector, subSolutionCoarseVector, positionCorrectorVector)
    correctVelocity!(subSolutionFineVector, subSolutionCoarseVector, velocityCorrectorVector)
end