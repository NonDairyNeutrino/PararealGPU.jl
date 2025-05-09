using LinearAlgebra: norm
using Statistics: mean

"""
    getrelativeChange(old_seq :: Vector{Vector{T}}, new_seq :: Vector{Vector{T}}) :: T where T <: AbstractFloat

Calculate the relative change between two vector sequences.
"""
function getRelativeChange(old_seq :: Vector{Vector{T}}, new_seq :: Vector{Vector{T}}) :: T where T <: AbstractFloat
    # indicator of relative change with arithmetic mean change
    # why arithmetic mean?  Cause it naturally extends to vectors, whereas others don't.
    # https://en.wikipedia.org/wiki/Relative_change#Indicators_of_relative_change
    disstanceVector = norm.(new_seq .- old_seq) 
    indicatorVector = norm.(mean([new_seq, old_seq]))
    relative_change = maximum(disstanceVector ./ indicatorVector)
    return relative_change
end

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
    # @assert all(!iszero, old_seq) "Old sequence contains $(count(iszero, old_seq)) zeros, first at $(findfirst(iszero, old_seq)). This is will cause a NaN."
    relativeChange      = getRelativeChange(old_seq, new_seq)
    @assert isfinite(relativeChange) "Some change is either NaN or infinite."
    iszero(relativeChange) && @warn "\nSolution did not change.  Be weary of these results."
    has_converged       = relativeChange <= threshold
    return has_converged, relativeChange
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
    position_has_converged, pos_change = hasConverged(oldSolution, newSolution, :positionSequence, threshold = threshold)
    velocity_has_converged, vel_change = hasConverged(oldSolution, newSolution, :velocitySequence, threshold = threshold)
    has_converged                      = position_has_converged && velocity_has_converged
    return has_converged, pos_change, vel_change
end
