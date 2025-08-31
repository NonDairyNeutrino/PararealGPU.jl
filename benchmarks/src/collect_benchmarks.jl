function get_all_coarse_benchmarks(dir :: String)
    file_name_vec          = readdir(dir)
    coarse_disc_string_vec = map(file_name_vec) do name
        m = match(r".*c(?<coarse>\d+)", name)
        isnothing(m) ? missing : m["coarse"]
    end
    x = findall(!ismissing, coarse_disc_string_vec)
    return hcat(file_name_vec[x], coarse_disc_string_vec[x])
end

function combine_benches!(dir :: String, out_file :: String)
    jldopen(out_file, "w") do file
        for (name, coarse) in eachrow(get_all_coarse_benchmarks(dir))
            fine_dict = load(joinpath(dir, name))
            for fine in keys(fine_dict)
                file[coarse * "/$fine"] = fine_dict[fine]
            end
        end
    end
    return out_file
end

function test()
    dir = "benchmarks/data/dist/"
    out = "benchmarks/data/bench_dist_test.jld2"
    combine_benches!(dir, out)
    bench = load(out; nested = true)
    return bench
end