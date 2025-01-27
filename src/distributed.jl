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
    devCount :: Int # or Vector{CuDevice} for slight performance increase
    function Host(name, pidVector, devVector)
        return new(name, pidVector[1], pidVector[2:end], devVector)
    end
end
function Base.show(io :: IO, host :: Host)
    println(io, "Host:   ", host.name)
    println(io, "Master:  ", host.master)
    println(io, "Workers: ", host.workerVector)
    println(io, "Devices: ", host.devCount)
end

"""
    spawnManagers(remoteHostNameVector :: Vector{String}) :: Vector{Int}

Spawn manager processes on each remote host.
"""
function spawnManagers(remoteHostNameVector :: Vector{String}) :: Vector{Int}
    println("Beginning with remote hosts: ", remoteHostNameVector)
    # create a worker process on each of remote hosts
    managerVector = addprocs(remoteHostNameVector) # TODO: port to use topology=:master_worker
    return managerVector
end

"""
    spawnWorkers(managerVector :: Vector{Int}) :: Vector{Tuple{String, Int}}

Spawn worker processes that will control device usage.
"""
function spawnWorkers(managerVector :: Vector{Int}) :: Vector{Tuple{String, Int}}
    @eval @everywhere using CUDA # load CUDA module on each process including master
    # hostDeviceCountVector
    hdcVector = pmap(_ -> (gethostname(), ndevices()), managerVector) # evals only on workers
    # spawn processes on remote hosts for each device
    addprocs(hdcVector)
    return hdcVector
end

"""
    createHostVector(remoteHostNameVector :: Vector{String}, managerVector :: Vector{Int}, devCountVector :: Vector{Int})

Bundle the process IDs with the number of devices on each remote host.
"""
function createHostVector(remoteHostNameVector :: Vector{String}, managerVector :: Vector{Int}, devCountVector :: Vector{Int})
    hostVector = similar(managerVector, Host)
    for i in eachindex(hostVector)
        name          = remoteHostNameVector[i]
        pidVector     = procs(managerVector[i]) # all pids on same machine as subMasterVector[i]
        devCount      = devCountVector[i] # devices are indexed at 0
        hostVector[i] = Host(name, pidVector, devCount)
    end
    println("The following hosts, procs, workers, and devices have been automatically recognized.")
    display(hostVector)
    return hostVector
end

desired result:
device 1 on host X to pid 2, 
device 2 on host X to pid 3, 
device 1 on host Y to pid 4, 
etc.
=#
# begin loop on master process
println("Beginning device assignment.")
for host in hostVector
    name      = host.name
    workerVector = host.workerVector
    devVector = host.devVector
    for (worker, dev) in zip(workerVector, devVector)
        println("      assigning device $dev to process $worker on host $name")
        # assign device to process pid
        remote_do(device!, worker, dev) # no fetch because device! returns nothing
    end
end

println("Confirming device assignemnt.")
@everywhere workers() println("proc ", myid(), " has device ", deviceid(device()), " on host ", gethostname())

rmprocs(workers())
