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
    manager    :: Int
    workerVector :: Vector{Int}
    devCount :: Int # or Vector{CuDevice} for slight performance increase
    function Host(name, pidVector, devCount)
        pidVector = 1 in pidVector ? pidVector[2:end] : pidVector
        return new(name, pidVector[1], pidVector[2:end], devCount)
    end
end
function Base.show(io :: IO, host :: Host)
    println(io, "Host:   ", host.name)
    println(io, "Manager:  ", host.manager)
    println(io, "Workers: ", host.workerVector)
    println(io, "Devices: ", host.devCount)
end

"""
    spawnManagers(remoteHostNameVector :: Vector{String}) :: Vector{Int}

Spawn manager processes on each remote host.
"""
function spawnManagers(remoteHostNameVector :: Vector{String}; addlocal :: Bool = false) :: Vector{Int}
    @info string("Beginning with ", addlocal ? "localhost and " : "", "remote hosts: ", remoteHostNameVector)
    # create a worker process on each of remote hosts
    if addlocal
        localManager    = addprocs(1, exeflags = `-t auto`)
    end
    remoteManagerVector = addprocs(remoteHostNameVector, exeflags = `-t auto`)
    managerVector       = addlocal ? [localManager; remoteManagerVector] : remoteManagerVector

    @info string("Loading PararealGPU.jl on all manager processes")
    @eval @everywhere workers() include("$(pwd())/src/PararealGPU.jl")
    @eval @everywhere workers() using .PararealGPU
    printstyled("PararealGPU.jl loaded on all manager processes\n", color=:green)
    return managerVector
end
# TODO: just use multiple threads per manager process and change the device for each thread
# then the managers asynchronously tell the gpus what to do
# syncrhonize
# send the director the results
# see https://cuda.juliagpu.org/stable/usage/multigpu/#Scenario-2:-Multiple-GPUs-per-process
getHDC()  = (gethostname(), ndevices())
getHDC(_) = getHDC()

"""
    spawnWorkers(managerVector :: Vector{Int}) :: Vector{Tuple{String, Int}}

Spawn worker processes that will control device usage.
"""
function spawnWorkers(managerVector :: Vector{Int}; addlocal = false) :: Vector{Tuple{String, Int}}
    @info string("Loading CUDA on all manager processes")
    @eval @everywhere workers() using CUDA # load CUDA module on each process including master
    printstyled("CUDA loaded on all manager processes\n", color=:green)

    # hostDeviceCountVector
    @info string("Getting number of devices on each host")
    hdcVector = pmap(getHDC, managerVector) # evals only on workers
    # spawn processes on remote hosts for each device
    @info string("Spawning processes for each device.")
    if addlocal
        localWorkers  = addprocs(hdcVector[1][2],  exeflags = `-t auto`)
        remoteWorkers = addprocs(hdcVector[2:end], exeflags = `-t auto`)
        deviceWorkers = [localWorkers; remoteWorkers]
    else
        deviceWorkers = addprocs(hdcVector, exeflags = `-t auto`)
    end

    @info string("Loading PararealGPU on each worker process")
    @eval @everywhere $deviceWorkers include("$(pwd())/src/PararealGPU.jl")
    @eval @everywhere $deviceWorkers using .PararealGPU
    printstyled("PararealGPU loaded on all worker processes.\n", color=:green)
    return hdcVector
end

"""
    createHostVector(remoteHostNameVector :: Vector{String}, managerVector :: Vector{Int}, devCountVector :: Vector{Int})

Bundle the process IDs with the number of devices on each remote host.
"""
function createHostVector(hdcVector :: Vector{Tuple{String, Int}}, managerVector :: Vector{Int})
    @info string("Collecting hosts, processes, and device counts")
    hostVector = similar(managerVector, Host)
    for i in eachindex(hostVector)
        name, devCount = hdcVector[i]
        pidVector      = procs(managerVector[i]) # all pids on same machine as subMasterVector[i]
        hostVector[i]  = Host(name, pidVector, devCount)
    end
    @info string("The following hosts, procs, workers, and devices have been automatically recognized.")
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
    @info string("Assigning devices to processes")
    @eval @everywhere using CUDA
    for host in hostVector
        name         = host.name
        workerVector = host.workerVector
        devIDVector  = 0:host.devCount-1
        for (worker, dev) in zip(workerVector, devIDVector)
            @info string("      Assigning device $dev to process $worker on host $name")
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
    @info string("Confirming device assignemnt")
    @everywhere workers() @info string("proc ", myid(), " has device ", deviceid(device()), " on host ", gethostname())
    return nothing
end

"""
    prepCluster(remoteHostNameVector :: Union{Vector{String}, Int}) :: CachingPool

Prepare a cluster and return a pool of device processes.
"""
function prepCluster(remoteHostNameVector :: Vector{String}; addlocal = false) :: Nothing
    managerVector  = spawnManagers(remoteHostNameVector, addlocal = addlocal)
    hdcVector      = spawnWorkers(managerVector;         addlocal = addlocal)
    # devCountVector = getindex.(hdcVector, 2)
    hostVector     = createHostVector(hdcVector, managerVector) # createHostVector(remoteHostNameVector, managerVector, devCountVector)
    assignDevices!(hostVector)
    showDeviceAssignments()

    devPool = getproperty.(hostVector, :workerVector) |> Iterators.flatten |> collect
    # @everywhere evaluate in the Main module, so need to explicitly put in PararealGPU module
    @everywhere PararealGPU.MANAGERPOOL = #= CachingPool =#$managerVector
    @everywhere PararealGPU.DEVPOOL     = #= CachingPool =#$devPool
    printstyled("Cluster created with MANAGERPOOL = $managerVector, DEVPOOL = $devPool\n", color=:green)
    return 
end