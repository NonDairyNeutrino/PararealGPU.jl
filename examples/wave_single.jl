# This is an example of using the Parareal algorithm to solve the simple
# initial value problem of d^2 u / dt^2 = -u with u(0) = u0, u'(0) = v0
# Author: Nathan Chapman
using Plots
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU
const nodeVector = String["Electromagnetism"]
prepCluster(nodeVector)

println("Creating initial value problems")
# DEFINE THE COARSE AND FINE PROPAGATION SCHEMES
const COARSEDISCRETIZATION = 8
const FINEDISCRETIZATION   = 8

const COARSEPROPAGATOR = Propagator(symplecticEuler, COARSEDISCRETIZATION) # how many problems / GPU cores
const FINEPROPAGATOR   = Propagator(velocityVerlet,  FINEDISCRETIZATION) # how many steps on each core / for each problem

@everywhere begin
"""
    acceleration(position :: Vector{T}, velocity :: Vector{T}) :: Vector{T} where T <: Real

Generate an acceleration function based on the given wave number.
"""
@inline function acceleration(position :: V, velocity :: V) :: V where V <: AbstractVector
    return -position
end
end

const INITIALPOSITION = [0.]
const INITIALVELOCITY = [1.]
const DOMAIN          = Interval(0., 2^1 * pi)
const IVP             = SecondOrderIVP("0", DOMAIN, acceleration, INITIALPOSITION, INITIALVELOCITY)

solution = solve(IVP, COARSEPROPAGATOR, FINEPROPAGATOR)
# TODO: write solutions to a file just in case something goes wrong after this

plot(sin, range(DOMAIN.lb, DOMAIN.ub, COARSEDISCRETIZATION + 1), label = "position - true")
plot!(
    solution.domain,
    [solution.positionSequence .|> first, solution.velocitySequence .|> first],
    label = ["position" "velocity"],
    title = "coarse: $COARSEDISCRETIZATION, fine: $FINEDISCRETIZATION"
)
println("Plot saved at ", pwd(), "/cos.png")
savefig("cos.png")
