"""
    Propagator(propagator :: Function, discretization :: Int)

Structured representation of a propagation scheme with a discretization.
"""
struct Propagator
    propagator :: Function
    discretization :: Int
end

"""
    Solution(domain :: Vector{Float64}, positionSequence :: Vector{Vector{Float64}}, velocitySequence :: Vector{Vector{Float64}})

Structured representation of the solution to a numerical differential equation.
"""
struct Solution
    domain           :: Vector         # time vector
    positionSequence :: Vector{Vector} # time vector of space vectors
    velocitySequence :: Vector{Vector} # time vector of space vectors
    function Solution(domain, positionSequence :: Vector{Vector}, velocitySequence = zeros(length(positionSequence) - 1) :: Vector{Vector})
        return new(domain, positionSequence, velocitySequence)
    end
end

"""
    propagate(ivp :: FirstOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Solution

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(ivp :: FirstOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Solution
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    solution          = similar(discretizedDomain, typeof(ivp.initialValue))

    solution[1]       = ivp.initialValue
    for i in Iterators.drop(eachindex(solution), 1)
        # broadcast propagator and derivative to allow for vector values
        solution[i] = propagator.propagator.(solution[i - 1], ivp.der.(discretizedDomain[i - 1], solution[i - 1]), step) + correctors[i]
    end
    return Solution(discretizedDomain, solution)
end

"""
    propagate(ivp :: SecondOrderIVP, propagator :: Propagator) :: Solution

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(ivp :: SecondOrderIVP, propagator :: Propagator) :: Solution
    step              = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain = discretize(ivp.domain, propagator.discretization)
    position          = similar(discretizedDomain, ivp.initialPosition |> typeof)
    velocity          = similar(discretizedDomain, ivp.initialVelocity |> typeof)

    position[1]       = ivp.initialPosition
    velocity[1]       = ivp.initialVelocity
    for i in Iterators.drop(eachindex(position), 1)
        position[i], velocity[i] = propagator.propagator(position[i - 1], velocity[i - 1], ivp.acceleration, step)
    end
    return Solution(discretizedDomain, position, velocity)
end

"""
    propagate(solver, acceleration, discretizedDomain, position, velocity) :: Tuple{Vector, Vector, Vector}

Propagates on the device.
"""
function propagate(solver, acceleration, discretizedDomain, position, velocity) :: Tuple{Vector, Vector, Vector}
    discretization = length(discretizedDomain)
    lowerBound     = discretizedDomain[1]
    upperBound     = discretizedDomain[end]
    step           = (upperBound - lowerBound) / discretization

    # define the discretized domain
    for i in 2:discretization
        discretizedDomain[i] = lowerBound + (i - 1) * step
    end

    # position evolution
    for i in 2:discretization
        position[i], velocity[i] = solver(position[i - 1], velocity[i - 1], acceleration, step)
    end

    return discretizedDomain, position, velocity
end

"""
    propagate(ivp :: SecondOrderIVP, propagator :: Propagator, correctors :: Vector{Float64} = zeros(propagator.discretization + 1)) :: Solution

Propagate an initial value problem using a given propagation scheme.
"""
function propagate(
    ivp :: SecondOrderIVP, 
    propagator :: Propagator, 
    positionCorrectorVector :: Vector{Vector{Float64}}, 
    velocityCorrectorVector :: Vector{Vector{Float64}}
    ) :: Solution
    step                = (ivp.domain.ub - ivp.domain.lb) / propagator.discretization
    discretizedDomain   = discretize(ivp.domain, propagator.discretization)
    positionSequence    = similar(discretizedDomain, ivp.initialPosition |> typeof)
    velocitySequence    = similar(discretizedDomain, ivp.initialVelocity |> typeof)

    positionSequence[1] = ivp.initialPosition
    velocitySequence[1] = ivp.initialVelocity
    for i in Iterators.drop(eachindex(positionSequence), 1)
        positionSequence[i], velocitySequence[i] = propagator.propagator(positionSequence[i - 1], velocitySequence[i - 1], ivp.acceleration, step)
        positionSequence[i] += positionCorrectorVector[i - 1]
        velocitySequence[i] += velocityCorrectorVector[i - 1]
    end
    return Solution(discretizedDomain, positionSequence, velocitySequence)
end
