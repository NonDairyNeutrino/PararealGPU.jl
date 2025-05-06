# This is an example of using the Parareal algorithm to solve the simple
# initial value problem of d^2 u / dt^2 = -u with u(0) = u0, u'(0) = v0
# Author: Nathan Chapman
using Plots
include("$(pwd())/src/PararealGPU.jl")
using .PararealGPU
const nodeVector = String["Electromagnetism"]
prepCluster(nodeVector, addlocal = true)

println("Creating initial value problems")
# DEFINE THE COARSE AND FINE PROPAGATION SCHEMES
const COARSEDISCRETIZATION = 2^16 # how many total problems
const FINEDISCRETIZATION   = 2^10 # each step is 1% of the domain # how many steps on each core

const COARSEPROPAGATOR = Propagator(symplecticEuler, COARSEDISCRETIZATION)
const FINEPROPAGATOR   = Propagator(velocityVerlet,  FINEDISCRETIZATION)

@everywhere begin
"""
    acceleration(position :: Vector{T}, velocity :: Vector{T}) :: Vector{T} where T <: Real

Generate an acceleration function based on the given wave number.
"""
@inline function acceleration(position :: V, velocity :: V) :: V where {T <: AbstractFloat, V <: AbstractVector{T}}
    k = 0.01f0 * pi
    return -k^2 * position
end
end

const INITIALPOSITION = Float32[0.]
const INITIALVELOCITY = Float32[1.]
const DOMAIN          = Interval{Float32}(0.0f0, 10.0f0)
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
plot_dir  = string(pwd(), "/examples/images/")
plot_name = "wave_single_$(COARSEDISCRETIZATION)_$FINEDISCRETIZATION.pdf"
plot_abs_path = plot_dir * plot_name
println("Plot saved at ", plot_abs_path)
savefig(plot_abs_path)
run(`codium $plot_abs_path`)
