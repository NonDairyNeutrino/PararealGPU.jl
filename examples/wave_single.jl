# This is an example of using the Parareal algorithm to solve the simple
# initial value problem of d^2 u / dt^2 = -u with u(0) = u0, u'(0) = v0
# Author: Nathan Chapman
using Plots
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU
const nodeVector = String["Electromagnetism"#= , "StrongForce" =#]
prepCluster(nodeVector, addlocal = true)

println("Creating initial value problems")
# DEFINE THE COARSE AND FINE PROPAGATION SCHEMES
const COARSEDISCRETIZATION = 2^10
const FINEDISCRETIZATION   = 100 # each step is 1% of the domain

const COARSEPROPAGATOR = Propagator(symplecticEuler, COARSEDISCRETIZATION) # how many problems / GPU cores
const FINEPROPAGATOR   = Propagator(velocityVerlet,  FINEDISCRETIZATION)   # how many steps on each core / for each problem

@everywhere begin
"""
    acceleration(position :: Vector{T}, velocity :: Vector{T}) :: Vector{T} where T <: Real

Generate an acceleration function based on the given wave number.
"""
@inline function acceleration(position :: V, velocity :: V) :: V where {T <: AbstractFloat, V <: AbstractVector{T}}
    return -position
end
end

const INITIALPOSITION = Float32[0.]
const INITIALVELOCITY = Float32[1.]
const DOMAIN          = Interval{Float32}(0, 2^2 * pi)
const IVP             = SecondOrderIVP("0", DOMAIN, acceleration, INITIALPOSITION, INITIALVELOCITY)

solution = solve(IVP, COARSEPROPAGATOR, FINEPROPAGATOR)
# TODO: write solutions to a file just in case something goes wrong after this

dom = range(DOMAIN.lb, DOMAIN.ub, COARSEDISCRETIZATION + 1)
plot(
    dom, 
    [sin.(dom) cos.(dom)],
    label = ["position - true" "velocity - true"]
)
plot!(
    solution.domain,
    [solution.positionSequence .|> first, solution.velocitySequence .|> first],
    label = ["position" "velocity"],
    title = "coarse: $COARSEDISCRETIZATION, fine: $FINEDISCRETIZATION"
)
plot_name = "cos_$(COARSEDISCRETIZATION)_$FINEDISCRETIZATION.pdf"
println("Plot saved at ", pwd(), "/", plot_name)
savefig(plot_name)
run(`codium $plot_name`)