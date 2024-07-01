function discretize(domain :: Interval, discretization :: Int) :: StepRangeLen
    return range(domain.lb, domain.ub, discretization + 1)
end

"""
    partition(domain :: Interval, discretization :: Int)

Partition an interval into a given number of mostly disjoint sub-domains.
"""
function partition(domain :: Interval, discretization :: Int) :: Vector{Interval}
    subdomains = Vector{Interval}(undef, COARSEDISCRETIZATION)
    step       = (domain.ub - domain.lb) / COARSEDISCRETIZATION
    for i in 0:COARSEDISCRETIZATION - 1
        subdomains[i + 1] = Interval(domain.lb + i * step, domain.lb + (i + 1) * step)
    end
    return subdomains
end