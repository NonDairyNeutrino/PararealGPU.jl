# This is an example of using the Parareal algorithm to solve the simple 
# initial value problem of du/dt = u with u(0) = u0
# Author: Nathan Chapman
# Date: 6/29/24
using Plots

# COMPUTATIONAL PARAMETERS
const COARSESTEP = 0.1
const FINESTEP   = COARSESTEP / 10
# PHYSICAL PARAMETERS
const DOMAIN           = Vector(1:2)
const INITIALVALUE = 1

"""
    der(t, u)

The differential equation represented as a definition of the derivative.
"""
function der(t, u)
    return u # functional representation of du/dt = u
end

"""
    euler(point, slope, step)

The Euler method of numerical integration.
"""
function euler(point, slope, step)
    return point + step * slope
end

"""
    coarse(point)

The coarse propagator.
"""
function coarse(point)
    return euler(point, der(point, point), COARSESTEP)
end

"""
    fine(point, slope)

The fine propagator
"""
function fine(point)
    return euler(point, der(point, point), FINESTEP)
end

# BEGIN ALGORITHM
# INITIALIZE
solution = similar(DOMAIN, Float64)
solution[1] = INITIALVALUE
for i in 2:length(DOMAIN)
    solution[i] = fine(solution[i-1])
end

display(solution)
plot(DOMAIN, [INITIALVALUE .+ exp.(DOMAIN), solution], label = ["Analytic" "Numeric"])