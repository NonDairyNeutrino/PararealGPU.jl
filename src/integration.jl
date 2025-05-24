# This is a collection of algorithms to numerically approximate the solution
# to a differential equation.

"""
    euler(point, slope, step)

The Euler method of numerical integration.
"""
function euler(point, slope, step)
    return point + step * slope
end

"""
    symplecticEuler(position :: Vector{T}, velocity :: Vector{T}, acceleration :: Function, timeStep :: R) :: Tuple{Vector{T}, Vector{T}} where {T <: Real, R <: Real}

Gives the single propagation using the symplectic Euler integrator.
"""
function symplecticEuler(position :: Vector{T}, velocity :: Vector{T}, acceleration :: Function, timeStep :: R) :: Tuple{Vector{T}, Vector{T}} where {T <: Real, R <: Real}
    positionNew = position + velocity * timeStep
    velocityNew = velocity + acceleration(positionNew, velocity) * timeStep
    return positionNew, velocityNew
end

"""
    velocityVerlet(
        position :: V,
        velocity :: V,
        acceleration :: Function,
        step :: T
    ) :: Tuple{V, V} where {T <: AbstractFloat, V <: Union{T, Vector{T}}}

Get the next position and velocity using the Velocity Verlet algorithm.
"""
function velocityVerlet(
        position :: V,
        velocity :: V,
        acceleration :: Function,
        step :: T
    ) :: Tuple{V, V} where {T <: AbstractFloat, V <: Union{T, Vector{T}}}
    halfstep        = convert(T, 0.5) * step
    halfstep2       = halfstep * step

    oldAcceleration = acceleration(position, velocity)
    newPosition     = position + velocity * step + halfstep2 * oldAcceleration
    newAcceleration = acceleration(newPosition, velocity)
    newVelocity     = velocity + halfstep * (oldAcceleration + newAcceleration)
    return newPosition, newVelocity
end