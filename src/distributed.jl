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

# get list of hosts
remoteHostVector = String["Electromagnetism"]
addprocs(remoteHostVector)

@everywhere using CUDA

# get list of devices on each host
hostDeviceCountVector = pmap(_ -> (gethostname(), CUDA.devices() |> length), workers())
hdcVector = hostDeviceCountVector # just an alias
display(hdcVector)

# spawn processes on remote hosts for each device
addprocs(hdcVector)
