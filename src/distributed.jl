# distributed functionality for PararealGPU.jl
using Distributed

#= 
The main idea is to:
on pid1
1. get list of hosts
2. get list of devices on each host
3. spawn processes on each host for each device pid2-pidND
4. assign a device to each process
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
    name      :: String
    master    :: Int
    workerVector :: Vector{Int}
    devVector :: Vector{Int} # or Vector{CuDevice} for slight performance increase
    function Host(name, pidVector, devVector)
        return new(name, pidVector[1], pidVector[2:end], devVector)
    end
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
    devVector     = 0:hdcVector[i][2]-1 |> collect # devices are indexed at 0
    hostVector[i] = Host(name, pidVector, devVector)
end

# distributed context to each processworkers
# @everywhere println("Loading from current directory: ", pwd())
@everywhere include("PararealGPU.jl")
@everywhere using .PararealGPU
# println("PararealGPU.jl successfully loaded on all processes.")

@everywhere @inline function acceleration(position :: Vector{T}, velocity :: Vector{T}, k = 1) :: Vector{T} where T <: Real
    return -k^2 * position # this encodes the differential equation u''(t) = -u
end
# println("acceleration loaded on all processes.")

#= 
assign device to each process
- master process (pid 1) which spawns and assigns ids to worker processes on remote hosts
(one for each device each host).  Each worker process knows its own id.
- So each host can have multiple worker processes 
e.g host X has pids 2, 3, and host Y has pids 4, 5, 6
- Then each process knows the devices (gpus) available to that _host_, but each device has its own
id _relative to the host_ e.g 
host X has pids 2, 3 and devices 1, 2, and 
host Y has pids 4, 5, 6 and devices 1, 2, 3

desired result:
device 1 on host X to pid 2, 
device 2 on host X to pid 3, 
device 1 on host Y to pid 4, 
etc.
=#
# begin loop on master process
for host in hostVector
    name      = host.name
    workerVector = host.workerVector
    devVector = host.devVector
    for (worker, dev) in zip(workerVector, devVector)
        println("assigning device $dev to process $worker on host $name")
        # assign device to process pid
        remote_do(device!, worker, dev) # no fetch because device! returns nothing
    end
end

@everywhere workers() println("proc ", myid(), " has device ", device(), " on host ", gethostname())

procs()[2:end] |> rmprocs
