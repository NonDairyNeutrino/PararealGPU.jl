"""
Parareal.jl provides functionality to solve an initial value problem in parallel through the use of a predictor-corrector scheme.
"""
module PararealGPU

export solve                                     # parareal.jl
export euler, symplecticEuler, velocityVerlet    # integration.jl

using CUDA
using Distributed

# see distributed.jl/prepCluster
MANAGERPOOL = nothing
DEVPOOL     = nothing

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
