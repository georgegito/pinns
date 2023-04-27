# This file is from julia documentation

using DifferentialEquations
using Plots

# Constants
α = 1.0

# Length of the domain (max x)
L = 1.0

# Max time
t_max = 0.2

# Discretize the spatial domain
Nx = 100
x = range(0, stop=L, length=Nx)

Nt = 100
dt = t_max / (Nt - 1)

# Compute the spatial step size
dx = x[2] - x[1]
# dx = L / (Nx - 1)

# Initial condition
u0 = sin.(π * x) + 0.5 * sin.(3 * π * x) + 0.25 * sin.(5 * π * x)

# PDE function
function heat_pde!(∂u∂t, u, p, t)
    α = p
    for i in 2:Nx-1
        ∂u∂t[i] = α * (u[i-1] - 2u[i] + u[i+1]) / (dx^2)
    end
    # Boundary conditions
    ∂u∂t[1] = 0
    ∂u∂t[end] = 0
end

# Time span
t_span = (0.0, t_max)

# Define the PDE problem
pde_problem = ODEProblem(heat_pde!, u0, t_span, α)

# Solve the PDE problem
pde_solution = solve(pde_problem, Tsit5(), saveat=dt)

# Collect the temperature field at each time step
# T = hcat([transpose(u) for u in pde_solution.u]...)

# Collect the temperature field at each time step
T = zeros(length(pde_solution.t), length(x))
for (i, t) in enumerate(pde_solution.t)
    T[i, :] = pde_solution(t)
end

# Create a contour plot
p = contour(x, pde_solution.t, T, xlabel="x", ylabel="t", title="Temperature Distribution", color=:turbo, levels=7, fill=true, lw=0)
display(p)