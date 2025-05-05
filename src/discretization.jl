"""
    discretize(domain :: Interval, discretization :: Int) :: StepRangeLen

TBW
"""
function discretize(domain :: Interval{T}, discretization :: Int) :: Vector{T} where T <: AbstractFloat
    return range(domain.lb, domain.ub, discretization + 1) |> collect
end

"""
    partition(domain :: Interval, discretization :: Int)

Partition an interval into a given number of mostly disjoint sub-domains.
"""
function partition(domain :: Interval{T}, discretization :: Int) :: Vector{Interval} where T <: AbstractFloat
    subdomains = Vector{Interval{T}}(undef, discretization)
    step       = (domain.ub - domain.lb) / discretization
    for i in 0:discretization - 1
        subdomains[i + 1] = Interval{T}(domain.lb + i * step, domain.lb + (i + 1) * step)
    end
    return subdomains
end