############# DICTIONARY INPUT #############

####### SINGLE DISC #######
"""
    dictslice(dict :: Dict, coarse :: Int, fine :: Colon) :: Dict

Get all fine benchmarks for a given coarse discretization
"""
function dictslice(dict :: Dict, coarse :: Int, fine :: Colon) :: Dict
    slice = dict["$coarse"]
    return slice
end

"""
    dictslice(dict :: Dict, coarse :: Colon, fine :: Int) :: Dict

Get all coarse benchmarks for a given fine discretization.
"""
function dictslice(dict :: Dict, coarse :: Colon, fine :: Int) :: Dict
    slice = Dict(k => v["$fine"] for (k, v) in dict)
    return slice
end

####### MULTIPLE DISC #######
"""
    dictslice(dict :: Dict, coarse :: Vector{Int}, fine :: Colon) :: Dict

Get all the benchmarks for a given coarse discretization.

    dictslice(dict :: Dict, coarse :: Colon, fine :: Vector{Int}) :: Dict

Get all the benchmarks for a given fine discretization.
"""
function dictslice(dict :: Dict, coarse :: Union{Vector{Int}, Colon}, fine :: Union{Vector{Int}, Colon}) :: Vector{Dict}
    slice_vec = dictslice.(Ref(dict), coarse, fine)
    return slice_vec
end

############# FILE INPUT #############
"""
    dictslice(dict_file :: String, coarse :: Int, fine :: Colon) :: Dict

Get all the benchmarks for a given coarse discretization.

    dictslice(dict_file :: String, coarse :: Colon, fine :: Int) :: Dict

Get all the benchmarks for a given fine discretization.
"""
function dictslice(dict_file :: String, coarse :: Union{Int, Vector{Int}, Colon}, fine :: Union{Int, Vector{Int}, Colon}) :: Union{Dict, Vector{Dict}}
    dict  = load(dict_file; nested = true)
    slice = dictslice(dict, coarse, fine)
    return slice
end
