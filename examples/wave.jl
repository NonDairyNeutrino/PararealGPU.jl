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
const COARSEPROPAGATOR = Propagator(symplecticEuler, 8192) # how many problems / GPU cores
const FINEPROPAGATOR   = Propagator(velocityVerlet,  2048) # how many steps on each core / for each problem


# @everywhere function acceleration(position :: Vector{T}, velocity :: Vector{T}; k = 1) :: Vector{T} where T <: Real
#     return -k^2 * position # this encodes the differential equation u''(t) = -u
# end

@everywhere function gen_acc(wave_number :: Int = 1)
    function acc(position :: Vector{T}, velocity :: Vector{T}) :: Vector{T} where T <: Real
        return -wave_number^2 * position
    end
    return acc
end

const INITIALPOSITION = [0.]
const INITIALVELOCITY = [1.]
const DOMAIN          = Interval(0., 2^1 * pi)

ivpVector = similar(devPool, SecondOrderIVP)
for k in 1:length(devPool)
    ivpVector[k] = SecondOrderIVP(DOMAIN, gen_acc(k), INITIALPOSITION, INITIALVELOCITY)
end

solutionVector = solve(COARSEPROPAGATOR, FINEPROPAGATOR, devPool, ivpVector)

for (k, rootSolution) in enumerate(solutionVector)
    plot!(
        rootSolution.domain,
        [rootSolution.positionSequence .|> first, rootSolution.velocitySequence .|> first],
        label = ["position - $k" "velocity - $k"],
        title = "discretization: $INITIALDISCRETIZATION    k <= $k"
    )
end
println("Plot saved at ", pwd(), "/cos.png")
savefig("cos.png")
