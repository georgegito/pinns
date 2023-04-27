using NeuralPDE, Lux, Optimization, OptimizationOptimJL
using CUDA, Random, ComponentArrays
import ModelingToolkit: Interval
# using OptimizationOptimisers # for Adam

use_gpu = false

@parameters t, x
@variables u(..)
Dxx = Differential(x)^2
Dtt = Differential(t)^2
Dt = Differential(t)

#PDE
alpha = 1
eq = Dt(u(t, x)) ~ alpha * Dxx(u(t, x))

# Initial and boundary conditions
bcs = [u(t, 0) ~ 0.0, # for all t > 0
    u(t, 1) ~ 0.0, # for all t > 0
    u(0, x) ~ (sin(π * x) + 0.5 * sin(3 * π * x) + 0.25 * sin(5 * π * x))] #for all  0 < x < 1

t_max = 0.25
x_max = 1.0

# Space and time domains
domains = [t ∈ Interval(0.0, t_max),
    x ∈ Interval(0.0, x_max)]

# Discretization parameters
Nx = 30  # Number of spatial grid points
Nt = 30  # Number of time steps
dx = x_max / (Nx - 1)
dt = t_max / (Nt - 1)

# Neural network
hidden_size = 16

chain = Chain(Dense(2, hidden_size, Lux.sigmoid_fast),
              Dense(hidden_size, hidden_size, Lux.sigmoid_fast),
              Dense(hidden_size, hidden_size, Lux.sigmoid_fast),
            #   Dense(hidden_size, hidden_size, Lux.sigmoid_fast),
            #   Dense(hidden_size, hidden_size, Lux.sigmoid_fast),
              Dense(hidden_size, 1))

if use_gpu
    ps = Lux.setup(Random.default_rng(), chain)[1]
    ps = ps |> ComponentArray |> Lux.gpu .|> Float64
end

# strategy = GridTraining([dt, dx])
Nf = Nx * Nt # number of collocation points for pde evalution
# Nb = Nx + 2 * Nt # number of points for boundary and initial conditions evaluation
Nb = Nx + 2 * Nt # number of points for boundary and initial conditions evaluation
# strategy = StochasticTraining(Nf + Nb, bcs_points=Nb)
# strategy = QuasiRandomTraining(Nf + Nb, bcs_points=Nb)
# strategy = QuasiRandomTraining(Nf + Nb, bcs_points=Nb, resampling=false, minibatch=1)
# strategy = QuasiRandomTraining(Nf + Nb)
strategy = QuasiRandomTraining(Nf + Nb, bcs_points=Nb, sampling_alg=NeuralPDE.SobolSample())
# strategy = QuasiRandomTraining(Nf + Nb, bcs_points=Nb, sampling_alg=NeuralPDE.SobolSample(), resampling=false, minibatch=1)
discretization = PhysicsInformedNN(chain, strategy)

@named pde_system = PDESystem(eq, bcs, domains, [t, x], [u(t, x)])
prob = discretize(pde_system, discretization)

epoch = 0

callback = function (p, l)
    global epoch
    epoch += 1
    if epoch % 10 == 0
        println("Epoch: $epoch\tLoss: $l")
    end
    return false
end

# optimizer
opt = Optim.BFGS()
# opt = Adam()
# opt = Optim.GradientDescent(P=0.01)
# res = @time Optimization.solve(prob, opt; callback=callback, maxiters=10000)
res = @time Optimization.solve(prob, opt; callback=callback, maxiters=10000)
phi = discretization.phi

using Plots

ts = [infimum(d.domain):0.1*dt:supremum(d.domain) for d in domains][1]
xs = [infimum(d.domain):0.1*dx:supremum(d.domain) for d in domains][2]

function analytic_sol_func(t, x, alpha)
    term1 = sin(π * x) * exp(-(π * alpha)^2 * t)
    term2 = 0.5 * sin(3 * π * x) * exp(-(9 * π * alpha)^2 * t)
    term3 = 0.25 * sin(5 * π * x) * exp(-(25 * π * alpha)^2 * t)

    T = term1 + term2 + term3
end

u_predict = reshape([first(phi([t, x], res.u)) for t in ts for x in xs],
                    (length(xs), length(ts)))
u_real = reshape([analytic_sol_func(t, x, alpha) for t in ts for x in xs],
                 (length(xs), length(ts)))

diff_u = abs.(u_predict .- u_real)

p1 = contour(xs, ts, u_real', xlabel="x", ylabel="t", title="Temperature Distribution - analytic", color=:turbo, levels=15, fill=true, lw=0)
p2 = contour(xs, ts, u_predict', xlabel="x", ylabel="t", title="Temperature Distribution - predict", color=:turbo, levels=15, fill=true, lw=0)
p3 = contour(xs, ts, diff_u', xlabel="x", ylabel="t", title="Error", color=:turbo, levels=15, fill=true, lw=0)

plot(p1, p2, p3, layout=(3, 1), size=(1000, 1500))