using Flux
using Random, NeuralPDE
using OrdinaryDiffEq, Optimisers, Statistics
import OptimizationOptimisers, OptimizationOptimJL

Random.seed!(100)

# Run a solve on scalars
linear = (u, p, t) -> cos(2pi * t)
tspan = (0.0f0, 1.0f0)
u0 = 0.0f0
prob = ODEProblem(linear, u0, tspan)
chain = Flux.Chain(Dense(1, 5, σ), Dense(5, 1))
opt = OptimizationOptimisers.Adam(0.1, (0.9, 0.95))

sol = solve(prob, NeuralPDE.NNODE(chain, opt), dt = 1 / 20.0f0, verbose = true,
            abstol = 1.0f-10, maxiters = 200)