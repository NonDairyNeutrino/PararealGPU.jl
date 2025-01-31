# distributed functionality for PararealGPU.jl
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
    function Host(name, pidVector, devCount)
        return new(name, pidVector[1], pidVector[2:end], devCount)
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
function spawnManagers(remoteHostNameVector :: Union{Vector{String}, Int}) :: Vector{Int}
    println("Beginning with remote hosts: ", remoteHostNameVector)
    # create a worker process on each of remote hosts
    managerVector = addprocs(remoteHostNameVector) # TODO: port to use topology=:master_worker

    println("Loading PararealGPU.jl on all manager processes")
    @eval @everywhere workers() include("$(pwd())/src/PararealGPU.jl")
    @eval @everywhere workers() using .PararealGPU
    println("PararealGPU.jl loaded on all manager processes")
    return managerVector
end

getHDC()  = (gethostname(), ndevices())
getHDC(_) = getHDC()

"""
    spawnWorkers(managerVector :: Vector{Int}) :: Vector{Tuple{String, Int}}

Spawn worker processes that will control device usage.
"""
function spawnWorkers(managerVector :: Vector{Int}) :: Vector{Tuple{String, Int}}
    println("Loading CUDA on all manager processes")
    @eval @everywhere workers() using CUDA # load CUDA module on each process including master
    println("CUDA loaded on all manager processes")

    # hostDeviceCountVector
    println("Getting number of devices on each host")
    hdcVector = pmap(getHDC, managerVector) # evals only on workers
    # spawn processes on remote hosts for each device
    println("spawning processes for each device.")
    addprocs(hdcVector)
    return hdcVector
end

"""
    createHostVector(remoteHostNameVector :: Vector{String}, managerVector :: Vector{Int}, devCountVector :: Vector{Int})

Bundle the process IDs with the number of devices on each remote host.
"""
function createHostVector(remoteHostNameVector :: Vector{String}, managerVector :: Vector{Int}, devCountVector :: Vector{Int})
    println("Collecting hosts, processes, and device counts")
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

"""
    assignDevices!(hostVector :: Vector{Host}) :: Nothing

Assign each device to a worker process on the same host.
"""
function assignDevices!(hostVector :: Vector{Host}) :: Nothing
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
    println("Assigning devices to processes")
    @eval @everywhere using CUDA
    for host in hostVector
        name         = host.name
        workerVector = host.workerVector
        devIDVector  = 0:host.devCount-1
        for (worker, dev) in zip(workerVector, devIDVector)
            println("      Assigning device $dev to process $worker on host $name")
            # assign device to process pid
            remote_do(device!, worker, dev)
        end
    end
    return nothing
end

"""
    showDeviceAssignments()

Show which device each process has for use.
"""
function showDeviceAssignments() :: Nothing
    println("Confirming device assignemnt")
    @everywhere workers() println("proc ", myid(), " has device ", deviceid(device()), " on host ", gethostname())
    return nothing
end

"""
    prepCluster(remoteHostNameVector :: Vector{String})

TBW
"""
function prepCluster(remoteHostNameVector :: Union{Vector{String}, Int})
    managerVector  = spawnManagers(remoteHostNameVector)
    hdcVector      = spawnWorkers(managerVector)
    devCountVector = getindex.(hdcVector, 2)
    hostVector     = createHostVector(remoteHostNameVector, managerVector, devCountVector)
    assignDevices!(hostVector)
    showDeviceAssignments()
end

