using NeuralPDE
using LinearAlgebra
using Plots
# using Flux
using Lux, Optimization
import OptimizationOptimisers
import ModelingToolkit: Interval
using CUDA, Random, ComponentArrays
using OptimizationOptimJL

use_gpu = false

@parameters t
@variables y(..)
Dt = Differential(t)
Dtt = Differential(t)^2

# Mass-spring-damper system parameters
m = 2.0
k = 20.0
ζ = 0.1  # Smaller damping ratio for slower energy dissipation
c = 2 * ζ * sqrt(m * k)

# ODE
f = (y, t) -> m * Dtt(y(t)) + c * Dt(y(t)) + k * y(t)
eq = f(y, t) ~ 0

# Time values for training data
t_u = range(0.0, stop=3.0, length=10)

# Angular frequency and damping ratio
ω0 = sqrt(k / m)
ω_d = ω0 * sqrt(1 - ζ^2)

# Calculate the exponential and oscillatory components
exp_term = exp.(-ζ * ω0 * t_u)
osc_term = cos.(ω_d * t_u) + (ζ / sqrt(1 - ζ^2)) * sin.(ω_d * t_u)

# Initial conditions
y0 = 1.0 # Initial displacement
v0 = 0.0 # Initial velocity

initial_conditions = [y(0.0) ~ y0, 
                    Dt(y(0.0)) ~ v0]
# ics = [y(0.0) ~ y0]

# Training points
y_u = y0 .* exp_term .* osc_term

training_points = [y(t_u_) ~ y_u_ for (t_u_, y_u_) in zip(t_u, y_u)]

# Boundary conditions
boundary_conditions = vcat(initial_conditions, training_points)

# Real system
t_real = range(0.0, stop=20.0, length=100)
ω0_real = sqrt(k / m)
ω_d_real = ω0 * sqrt(1 - ζ^2)
exp_term_real = exp.(-ζ * ω0_real * t_real)
osc_term_real = cos.(ω_d_real * t_real) + (ζ / sqrt(1 - ζ^2)) * sin.(ω_d_real * t_real)
y_real = y0 .* exp_term_real .* osc_term_real

p1 = plot(t_real, y_real, xlabel="t", ylabel="y", label="Real system output", linewidth=:1.5)
p1 = scatter!(t_u, y_u, label="Training points")
# display(p1)

# Solve
# tspan = (0.0, 20.0)
# prob = ODEProblem(f, ics, tspan)
chain = Lux.Chain(Lux.Dense(1, 64, Lux.σ), 
                Lux.Dense(64, 64, Lux.σ),
                # Lux.Dense(32, 32, Lux.σ),
                Lux.Dense(64, 1))

if use_gpu
    ps = Lux.setup(Random.default_rng(), chain)[1]
    ps = ps |> ComponentArray |> Lux.gpu .|> Float64
end

# opt = OptimizationOptimisers.Adam(0.01)
# opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))
opt = Optim.BFGS()

# sol = solve(prob, NeuralPDE.NNODE(chain, opt))
# sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt = 1 / 20.0f0, verbose = true,
            # abstol = 1.0f-10, maxiters = 200)

# t_f = range(0, stop=20, length=100) # collocation points

# strategy = GridTraining(0.1)
strategy = QuasiRandomTraining(200, bcs_points=length(boundary_conditions)/2)
discretization = PhysicsInformedNN(chain, strategy)

domain = [t ∈ Interval(0.0, 20.0)]

@named system = PDESystem(eq, boundary_conditions, domain, [t], [y(t)])
prob = discretize(system, discretization)

epoch = 0

callback = function (p, l)
    global epoch
    epoch += 1
    if epoch % 100 == 0
        println("Epoch: $epoch\tLoss: $l")
    end
    return false
end

res = @time Optimization.solve(prob, opt; callback=callback, maxiters=1000)
phi = discretization.phi

# ts = [infimum(d.domain):0.01:supremum(d.domain) for d in domain]
t_span = (0.0:0.01:20.0)
y_predict = [first(phi([t], res.u)) for t in t_span]

p1 = plot!(t_span, y_predict, xlabel="t", ylabel="y", label="PINN prediction", linewidth=:1.5, linecolor=:red, linestyle=:dot)
plot(p1)
# plot(p1, p2, layout=(2, 1), size=(500, 500))

# p3 = p1
# p3 = plot!(t_span, y_predict, xlabel="t", ylabel="y", label="PINN prediction")