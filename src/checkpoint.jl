# write current iteration's solution to checkpoint file
# each iteration overwrites the previous checkpoint

"""
    write_checkpoint(path :: String = "solution_checkpoint.txt") :: Nothing

Write the solution at the current iteration to a file.
"""
function write_checkpoint(solution :: Solution, path :: String = "solution_checkpoint.txt") :: Nothing
    @info "Creating solution checkpoint file"
    writedlm(
        path, 
        vcat(
            permutedims(solution.domain), 
            stack(solution.positionSequence), 
            stack(solution.velocitySequence)
        )
    )
    return nothing
end

function read_checkpoint(path :: String, T :: Type) :: Solution
    mat   = readdlm(path, T)
    nrows = size(mat, 1)
    # there are 2n + 1 rows because
    # the first row is the domain
    # the second row is the x-position
    # ...
    # the first row after the position is the x-velocity
    # ...
    # so the dimension of the position and velocity is (nrows - 1)/2
    dim       = div((nrows - 1), 2)
    pos_start = 2
    pos_end   = pos_start + dim - 1
    vel_start = pos_end + 1
    # vel_end   = end

    domain           = mat[1, :]
    positionSequence = mat[pos_start:pos_end, :] |> eachcol .|> collect
    velocitySequence = mat[vel_start:end, :]     |> eachcol .|> collect
    solution         = Solution(domain, positionSequence, velocitySequence)
    return solution
end