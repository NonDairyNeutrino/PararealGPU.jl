"""
    hasConverged(
        old_sol :: Solution{T}, 
        new_sol :: Solution{T}, 
        seq     :: Symbol; 
        threshold    :: T = sqrt(eps(T))
    ) :: Tuple{Bool, T} where T <: AbstractFloat

Determines convergence of the given property given the previous and and current solutions.
"""
function hasConverged(
        old_sol :: Solution{T}, 
        new_sol :: Solution{T}, 
        seq :: Symbol;
        threshold :: T = sqrt(eps(T))
    ) :: Bool where T <: AbstractFloat
    # the initial condition never changes so just skip it
    old_seq             = getproperty(old_sol, seq)[2:end]
    new_seq             = getproperty(new_sol, seq)[2:end]
    # @assert all(!iszero, old_seq) "Old sequence contains $(count(iszero, old_seq)) zeros, first at $(findfirst(iszero, old_seq)). This is will cause a NaN."

    #= 
    I trust the people are smarter than me and so I should just use isapprox instead of choosing my
    own comparison method.

    Docstring for isapprox
        isapprox(x, y; atol::Real=0, rtol::Real=atol>0 ? 0 : √eps, nans::Bool=false[, norm::Function])
        isapprox returns true if norm(x-y) <= max(atol, rtol*max(norm(x), norm(y)))

    sqrt(eps(Float32)) == 0.00034526698f0
    sqrt(eps(Float64)) == 1.4901161193847656e-8

    The norm of a vector of vectors is the norm of the vector of norms i.e.
    norm(v :: Vector{Vector}) == LinearAlgebra.norm(norm.(v))
    =#
    has_converged       = isapprox(old_seq, new_seq, rtol=threshold) #= relativeChange <= threshold =#
    return has_converged
end

"""
    hasConverged(
        oldSolution :: Solution{T}, 
        newSolution :: Solution{T}; 
        threshold        :: T = sqrt(eps(T))
    ) :: Bool where T <: AbstractFloat

Determines convergence of both position and velocity given the previous and current solutions.
"""
function hasConverged(
        oldSolution :: Solution{T}, 
        newSolution :: Solution{T}; 
        threshold        :: T = sqrt(eps(T))
    ) :: Bool where T <: AbstractFloat
    position_has_converged = hasConverged(oldSolution, newSolution, :positionSequence, threshold = threshold)
    velocity_has_converged = hasConverged(oldSolution, newSolution, :velocitySequence, threshold = threshold)
    has_converged          = position_has_converged && velocity_has_converged
    return has_converged
end
