# distributed functionality for PararealGPU.jl
using Distributed, DistributedEnvironments

#= 
The main idea is to:
on pid1
1. get list of hosts
2. get list of devices on each host
3. spawn processes on each host for each device pid2-pidND
4. @everywhere begin
    include(PararealGPU.jl)
    using .Parareal
    function acceleration(...) ... end
   end
5. create list of IVPs for each wave number
6. 
    solutionVector = pmap(ivp -> parareal(...), ivpVector)
=#

# get list of hosts
localHost        = gethostname()
remoteHostVector = String["Electromagnetism"#= , "Gravity" =#]
@initcluster remoteHostVector

# addprocs(1)                # create a worker process on the local host
addprocs(remoteHostVector) # create a worker process on each of remote hosts

@everywhere using CUDA

# get list of devices on each host
hostDeviceCountVector = pmap(_ -> (gethostname(), CUDA.devices() |> length), workers())
hdcVector = hostDeviceCountVector # just an alias
display(hdcVector)

# spawn processes on remote hosts for each device
addprocs(hdcVector)

@everywhere println("This is a message from machine ", gethostname(), " on process ", myid())

# distributed context to each process
@everywhere begin
    include("PararealGPU.jl")
    using .PararealGPU
    function acceleration(position :: Vector{T}, velocity :: Vector{T}, k :: T = 1) :: Vector{T} where T <: Real
        return -k^2 * position # this encodes the differential equation u''(t) = -u
    end
end

@everywhere names(PararealGPU)
