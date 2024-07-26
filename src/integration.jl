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
    verlet(position :: Vector{Float64}, prev_position :: Vector{Float64}, acceleration :: Vector{Float64}, step :: Float64) :: Vector{Float64}

Get the next position given the current and previous positions, and the current acceleration.

NOTE: This is the base Verlet algorithm, not Stormer-Verlet nor Velocity-Verlet.
"""
function verlet(position :: Vector{Float64}, prev_position :: Vector{Float64}, acceleration :: Vector{Float64}, step :: Float64) :: Vector{Float64}
    return 2 * position - prev_position + acceleration * step^2
end

function velocityVerlet(position :: Vector{Float64}, velocity :: Vector{Float64}, step :: Float64)
    
end