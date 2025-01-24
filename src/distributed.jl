# distributed functionality for PararealGPU.jl
using Distributed

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
struct Host
    name      :: String# or Vector{CuDevice} for slight performance increase
    devVector :: Vector{Int} # or Vector{CuDevice} for slight performance increase
end

remoteHostNameVector = String["Electromagnetism"]
# create a worker process on each of remote hosts
subMasterVector = addprocs(remoteHostNameVector) # TODO: port to use topology=:master_worker

@everywhere using CUDA # load CUDA module on each process including master
# hostDeviceCountVector
hdcVector = pmap(_ -> (gethostname(), ndevices()), subMasterVector) # evals only on workers
# spawn processes on remote hosts for each device
addprocs(hdcVector)
@everywhere using CUDA

hostVector = similar(subMasterVector, Host)
for i in eachindex(hostVector)
    name          = remoteHostNameVector[i]
    pidVector     = procs(subMasterVector[i]) # all pids on same machine as subMasterVector[i]
    devVector     = 1:hdcVector[i][2] |> collect
    hostVector[i] = Host(name, pidVector, devVector)
end

# distributed context to each process
@everywhere begin
    include("PararealGPU.jl")
    using .PararealGPU
    function acceleration(position :: Vector{T}, velocity :: Vector{T}, k :: T = 1) :: Vector{T} where T <: Real
        return -k^2 * position # this encodes the differential equation u''(t) = -u
    end
end

@everywhere names(PararealGPU)
