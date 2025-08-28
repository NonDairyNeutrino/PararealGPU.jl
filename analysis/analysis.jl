"""
    calculate_potential_energy(angle :: Vector{T}; mass = 1, gravity = 1, rod_length = 1) :: Float64 where T <: AbstractFloat

Calculate the potential energy of the pendulum at a given position.
"""
function calculate_potential_energy(angle :: Vector{T}; mass = 1, gravity = 1, rod_length = 1) :: Float64 where T <: AbstractFloat
    if length(angle) == 1
        theta = angle[1]
    else
        theta, phi = angle
    end
    height           = rod_length * (1 - cos(theta))
    potential_energy = mass * gravity * height
    return potential_energy
end

"""
    calculate_potential_energy(solution; gravity = 1, rod_length = 1) :: Float64

Calculate the potential energy of the final state of the pendulum.
"""
function calculate_potential_energy(solution; mass = 1, gravity = 1, rod_length = 1) :: Float64
    final_angle      = solution.positionSequence |> last
    potential_energy = calculate_potential_energy(final_angle; mass = mass, gravity = gravity, rod_length = rod_length)
    return potential_energy
end

"""
    calculate_kinetic_energy(velocity :: Vector{T}; mass = 1) :: Float64 where T <: AbstractFloat

Calculate the kinetic energy of the pendulum for a given velocity.
"""
function calculate_kinetic_energy(velocity :: Vector{T}; mass = 1) :: Float64 where T <: AbstractFloat
    kinetic_energy = (1//2) * mass * sum(abs2, velocity)
    return kinetic_energy
end

"""
    calculate_kinetic_energy(solution; mass = 1) :: Float64

Calculate the kinetic energy of the final state of the pendulum.
"""
function calculate_kinetic_energy(solution; mass = 1) :: Float64
    final_velocity = solution.velocitySequence |> last
    kinetic_energy = calculate_kinetic_energy(final_velocity; mass = mass)
    return kinetic_energy
end

"""
    calculate_mechanical_energy(position :: Vector{T}, velocity :: Vector{T}; mass = 1, gravity = 1, rod_length = 1) :: Float64 where T <: AbstractFloat

Calculate the total mechanical energy of the pendulum for a given position and velocity.
"""
function calculate_mechanical_energy(position :: Vector{T}, velocity :: Vector{T}; mass = 1, gravity = 1, rod_length = 1) :: Float64 where T <: AbstractFloat
    ke = calculate_kinetic_energy(velocity; mass = mass)
    pe = calculate_potential_energy(position; mass = mass, gravity = gravity, rod_length = rod_length)
    me = ke + pe
    return me
end

"""
    calculate_energy(solution) :: Float64

Calculate the total mechanical energy of the final state of the pendulum.
"""
function calculate_mechanical_energy(solution; mass = 1, gravity = 1, rod_length = 1) :: Float64
    pe = calculate_potential_energy(solution; mass = mass, gravity = gravity, rod_length = rod_length)
    ke = calculate_kinetic_energy(solution; mass = mass)
    me = ke + pe # mechanical energy

    #= 
    an alternative implementation could be

    final_position = solution.positionSequence |> last
    final_velocity = solution.velocitySequence |> last
    me = calculate_mechanical_energy(final_position, final_velocity; kwargs...)
    =#

    return me
end

"""
    calculate_error(initial_energy :: Float64, solution) :: Float64

Calculate the relative energy drift of the pendulum at its final state.
"""
function calculate_error(initial_energy :: Float64, solution; mass = 1, gravity = 1, rod_length = 1) :: Float64
    energy = calculate_mechanical_energy(solution; mass = mass, gravity = gravity, rod_length = rod_length)
    relative_error = (energy - initial_energy) / initial_energy
    return relative_error
end

"""
    calculate_error(initial_energy :: Float64) :: Function

Create a closure to calculate error for any solution compared to a given initial energy.
"""
calculate_error(initial_energy :: Float64) :: Function = Base.Fix1(calculate_error, initial_energy)

"""
    calculate_efficiency(initial_energy :: Float64, bench) :: Float64

Calculate the efficiency of a simulation.
"""
function calculate_efficiency(initial_energy :: Float64, bench; mass = 1, gravity = 1, rod_length = 1) :: Float64
    sol, iters = bench.value
    runtime    = bench.time
    err = calculate_error(initial_energy, sol; mass = mass, gravity = gravity, rod_length = rod_length)
    eff = err * runtime
    return eff
end

"""
    compare_efficiency(initial_energy :: Float64, bench_seq, bench_par) :: Float64

Measure the efficiency of a parallel method by comparing its error and runtime to that of sequential.
"""
function compare_efficiency(initial_energy :: Float64, bench_seq, bench_par; mass = 1, gravity = 1, rod_length = 1) :: Float64
    seq_eff = calculate_efficiency(initial_energy, bench_seq; mass = mass, gravity = gravity, rod_length = rod_length)
    par_eff = calculate_efficiency(initial_energy, bench_par; mass = mass, gravity = gravity, rod_length = rod_length)
    eff     = par_eff / seq_eff
    return eff
end

function load_bench(bench_file :: String, coarse :: Int, fine :: Int)
    bench_dist = load(bench_file; nested = true)
    bench      = bench_dist["$coarse"]["$fine"]
    return bench
end

# for method in ["single", "gpu", "dist"]
#     bench_method_name = "bench_" * method * ".jld2"
#     bench_method      = jldopen(bench_method_name)
#     for coarse in 3:14
#         bench_coarse = bench_method["$coarse"]
#         for fine in 3:14
#             bench = method == "single" ? bench_coarse : bench_coarse["$fine"]
#             bench.value[1].domain |> display
#         end
#     end
# end