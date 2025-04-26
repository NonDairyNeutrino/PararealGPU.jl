# This is an example of using the Parareal algorithm to solve the simple
# initial value problem of d^2 u / dt^2 = -u with u(0) = u0, u'(0) = v0
# Author: Nathan Chapman
using Plots
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU
const nodeVector = String["Electromagnetism"]
devPool = prepCluster(nodeVector)

println("Creating initial value problems")
# DEFINE THE COARSE AND FINE PROPAGATION SCHEMES
const COARSEDISCRETIZATION = 8
const FINEDISCRETIZATION   = 2048
# TODO: change Propagator to closures
const COARSEPROPAGATOR = Propagator(symplecticEuler, COARSEDISCRETIZATION) # how many problems / GPU cores
const FINEPROPAGATOR   = Propagator(velocityVerlet,  FINEDISCRETIZATION) # how many steps on each core / for each problem

@everywhere begin
"""
    gen_acc(wave_number :: Int = 1) :: Function

Generate an acceleration function based on the given wave number.
"""
function gen_acc(wave_number :: Int = 1) :: Function
    function acc(position :: Vector{T}, velocity :: Vector{T}) :: Vector{T} where T <: Real
        return -wave_number^2 * position
    end
    return acc
end
end

const INITIALPOSITION = [0.]
const INITIALVELOCITY = [1.]
const DOMAIN          = Interval(0., 2^1 * pi)

ivpVector = Vector{SecondOrderIVP}(undef, length(devPool))
for k in 1:length(devPool)
    ivpVector[k] = SecondOrderIVP(DOMAIN, gen_acc(k), INITIALPOSITION, INITIALVELOCITY)
end

solutionVector = solve(COARSEPROPAGATOR, FINEPROPAGATOR, devPool, ivpVector)
# TODO: write solutions to a file just in case something goes wrong after this

plot(sin, range(DOMAIN.lb, DOMAIN.ub, COARSEDISCRETIZATION + 1), label = "position - true")
for (k, rootSolution) in enumerate(solutionVector)
    plot!(
        rootSolution.domain,
        [rootSolution.positionSequence .|> first, rootSolution.velocitySequence .|> first],
        label = ["position - $k" "velocity - $k"],
        title = "coarse: $COARSEDISCRETIZATION, fine: $FINEDISCRETIZATION"
    )
end
println("Plot saved at ", pwd(), "/cos.png")
savefig("cos.png")
