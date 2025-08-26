"""
    calculate_potential_energy(solution :: Solution; gravity = 1, rod_length = 1) :: Float64

Calculate the potential energy of the final state of the pendulum.
"""
function calculate_potential_energy(solution :: Solution; gravity = 1, rod_length = 1) :: Float64
    final_position   = solution.positionSequence |> last
    potential_energy = gravity * rod_length * (1 - cos(final_position))
    return potential_energy
end

"""
    calculate_kinetic_energy(solution :: Solution; mass = 1) :: Float64

Calculate the kinetic energy of the final state of the pendulum.
"""
function calculate_kinetic_energy(solution :: Solution; mass = 1) :: Float64
    final_velocity = solution.velocitySequence |> last
    kinetic_energy = (1//2) * mass * sum(abs2, final_velocity)
    return kinetic_energy
end

"""
    calculate_energy(solution :: Solution) :: Float64

Calculate the total mechanical energy of the final state of the pendulum.
"""
function calculate_energy(solution :: Solution) :: Float64
    pe = calculate_potential_energy(solution)
    ke = calculate_kinetic_energy(solution)
    me = ke + pe # mechanical energy
    return me
end

"""
    calculate_error(initial_energy :: Float64, solution :: Solution) :: Float64

Calculate the relative energy drift of the pendulum at its final state.
"""
function calculate_error(initial_energy :: Float64, solution :: Solution) :: Float64
    energy = calculate_energy(solution)
    relative_error = (energy - initial_energy) / initial_energy
    return relative_error
end

"""
    calculate_error(initial_energy :: Float64) :: Function

Create a closure to calculate error for any solution compared to a given initial energy.
"""
calculate_error(initial_energy :: Float64) :: Function = Base.Fix1(calculate_error, initial_energy)

"""
    calculate_efficiency(initial_energy :: Float64, bench :: NamedTuple) :: Float64

Calculate the efficiency of a simulation.
"""
function calculate_efficiency(initial_energy :: Float64, bench :: NamedTuple) :: Float64
    sol, iters = bench.value
    runtime    = bench.time
    err = calculate_error(initial_energy, sol)
    eff = err * runtime
    return eff
end

"""
    calculate_efficiency(bench_seq :: NamedTuple, bench_par :: NamedTuple) :: Float64

Measure the efficiency of a parallel method by comparing its error and runtime to that of sequential.
"""
function compare_efficiency(initial_energy :: Float64, bench_seq :: NamedTuple, bench_par :: NamedTuple) :: Float64
    seq_eff = calculate_efficiency(initial_energy, bench_seq)
    par_eff = calculate_efficiency(initial_energy, bench_par)
    eff = par_eff / seq_eff
    return eff
end

