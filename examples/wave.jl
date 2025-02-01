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
const INITIALDISCRETIZATION = Threads.nthreads()
const COARSEPROPAGATOR      = Propagator(symplecticEuler, INITIALDISCRETIZATION)
const FINEPROPAGATOR        = Propagator(velocityVerlet,  2^0 * INITIALDISCRETIZATION)

@everywhere @inline function acceleration(position :: Vector{T}, velocity :: Vector{T}; k = 1) :: Vector{T} where T <: Real
    return -k^2 * position # this encodes the differential equation u''(t) = -u
end
const INITIALPOSITION = [0.]
const INITIALVELOCITY = [1.]
const DOMAIN          = Interval(0., 2^2 * pi)
# const IVP             = SecondOrderIVP(DOMAIN, acceleration, INITIALPOSITION, INITIALVELOCITY) # second order initial value problem

ivpVector = Vector{SecondOrderIVP}(undef, length(devPool))
for k in 1:length(devPool)
    ivpVector[k] = SecondOrderIVP(DOMAIN, (x, v) -> acceleration(x, v; k), INITIALPOSITION, INITIALVELOCITY)
end

println("Beginning parareal evaluation on workers")
solutionVector = pmap(ivp -> parareal(ivp, COARSEPROPAGATOR, FINEPROPAGATOR), devPool, ivpVector)
println("Parareal evaluation finished")

for (k, rootSolution) in enumerate(solutionVector)
    plot!(
        rootSolution.domain,
        [rootSolution.positionSequence .|> first, rootSolution.velocitySequence .|> first],
        label = ["position" "velocity"],
        title = "k = $k"
    )
end
println("Plot saved at ", pwd(), "/cos.png")
savefig("cos.png")
