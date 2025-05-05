"""
    correct!(
        subSolutionFineVector   :: Vector{Solution{T}}, 
        subSolutionCoarseVector :: Vector{Solution{T}},
        positionCorrectorVector :: Vector{Vector{T}},
        velocityCorrectorVector :: Vector{Vector{T}}
    ) :: Nothing where T <: AbstractFloat

Correct solutions positions and velocities.
"""
function correct!(
        subSolutionFineVector   :: Vector{Solution{T}}, 
        subSolutionCoarseVector :: Vector{Solution{T}},
        positionCorrectorVector :: Vector{Vector{T}},
        velocityCorrectorVector :: Vector{Vector{T}}
    ) :: Nothing where T <: AbstractFloat

    fineCorrectorVector     = getproperty.(subSolutionFineVector,   :positionSequence) .|> last
    coarseCorrectorVector   = getproperty.(subSolutionCoarseVector, :positionSequence) .|> last
    positionCorrectorVector .= fineCorrectorVector .- coarseCorrectorVector

    fineCorrectorVector     = getproperty.(subSolutionFineVector,   :velocitySequence) .|> last
    coarseCorrectorVector   = getproperty.(subSolutionCoarseVector, :velocitySequence) .|> last
    velocityCorrectorVector .= fineCorrectorVector .- coarseCorrectorVector
    return nothing
end
