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
    sympecticEuler(timeStep :: Float64, position :: Vector{Float64}, velocity :: Vector{Float64}, acceleration :: Function) :: Tuple{Vector{Float64}, Vector{Float64}}

Gives the single propagation using the symplectic Euler integrator.
"""
function symplecticEuler(position :: Vector{T}, velocity :: Vector{T}, acceleration :: Function, timeStep :: R) :: Tuple{Vector{T}, Vector{T}} where {T <: Real, R <: Real}
    positionNew = position + velocity * timeStep
    velocityNew = velocity + acceleration(positionNew, velocity) * timeStep
    return positionNew, velocityNew
end

"""
    velocityVerlet(position :: Vector{Float64}, velocity :: Vector{Float64}, acceleration :: Function, step :: Float64) :: Vector{Float64}

Get the next position and velocity using the Velocity Verlet algorithm.
"""
function velocityVerlet(position :: Vector{T}, velocity :: Vector{T}, acceleration :: Function, step :: Float64) :: Vector{Vector{T}} where T <: Real
    oldAcceleration = acceleration(position, velocity)
    newPosition     = position + velocity * step + 0.5 * oldAcceleration * step^2
    newAcceleration = acceleration(newPosition, velocity)
    newVelocity     = velocity + 0.5 * (oldAcceleration + newAcceleration) * step
    return [newPosition, newVelocity]
end