"""
Parareal.jl provides functionality to solve an initial value problem in parallel through the use of a predictor-corrector scheme.
"""
module PararealGPU

export euler, symplecticEuler, velocityVerlet  # integration.jl
export Interval, FirstOrderIVP, SecondOrderIVP # ivp.jl
export Propagator
export prepCluster, getHDC                     # distributed.jl
export parareal                                # Parareal.jl
export @everywhere, pmap, myid                 # Distributed

using CUDA
# using Adapt: @adapt_structure
using LinearAlgebra: norm
using Distributed

include("ivp.jl")
include("discretization.jl")
include("propagation.jl")
include("subproblems.jl")
include("integration.jl")
include("correction.jl")
include("convergence.jl")
include("kernel.jl")
include("distributed.jl")
include("parareal.jl")

end
