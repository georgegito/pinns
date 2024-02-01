using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, QuasiMonteCarlo, LineSearches
import ModelingToolkit: Interval, infimum, supremum

@parameters x, y, z
@variables u(..), v(..), w(..), p(..)
Dx = Differential(x)
Dy = Differential(y)
Dz = Differential(z)
Dxx = Dx^2
Dyy = Dy^2
Dzz = Dz^2

ρ = 1.225
μ = 1.81e-5

x_min = 0
y_min = 0
z_min = 0
x_max = 1
y_max = 1
z_max = 1

u_func = u(x, y, z)
v_func = v(x, y, z)
w_func = w(x, y, z)
p_func = p(x, y, z)

continuity_eq = Dx(u_func) + Dy(v_func) + Dz(w_func) ~ 0
momentum_eq_x = u_func*Dx(u_func) + v_func*Dy(u_func) + w_func*Dz(u_func) ~ -1/ρ*Dx(p_func) + μ*(Dxx(u_func) + Dyy(u_func) + Dzz(u_func))
momentum_eq_y = u_func*Dx(v_func) + v_func*Dy(v_func) + w_func*Dz(v_func) ~ -1/ρ*Dy(p_func) + μ*(Dxx(v_func) + Dyy(v_func) + Dzz(v_func))
momentum_eq_z = u_func*Dx(w_func) + v_func*Dy(w_func) + w_func*Dz(w_func) ~ -1/ρ*Dz(p_func) + μ*(Dxx(w_func) + Dyy(w_func) + Dzz(w_func))

# TODO: poisson

eqs = [continuity_eq, momentum_eq_x, momentum_eq_y, momentum_eq_z]

u_in = 0
v_in = -1
w_in = 0

u_right = 0
u_left = 0

w_up = 0

u_down = 0
v_down = 0
w_down = 0

p_out = 1

bcs = [
  # inflow bc
  u(x, y_max, z) ~ u_in, 
  v(x, y_max, z) ~ v_in, 
  w(x, y_max, z) ~ w_in,

  # outflow bc
  p(x, y_min, z) ~ p_out,

  # left bc
  u(x_min, y, z) ~ u_left,

  # right bc
  u(x_max, y, z) ~ u_right,

  # down bc
  u(x, y, z_min) ~ u_down,
  v(x, y, z_min) ~ v_down,
  w(x, y, z_min) ~ w_down,

  # up bc
  w(x, y, z_max) ~ w_up
]

# Space and time domains
domains = 
  [x ∈ Interval(x_min, x_max),
    y ∈ Interval(y_min, y_max),
    z ∈ Interval(z_min, z_max)]

# Neural network
input_size = 3
output_size = 4

hidden_layer_size = 100
num_hidden_layers = 3

hidden_layers = [Dense(hidden_layer_size, hidden_layer_size, Lux.tanh) for _ in 1:num_hidden_layers]

chain = Lux.Chain(Dense(input_size, hidden_layer_size, Lux.tanh), hidden_layers, Dense(hidden_layer_size, output_size))

Nf = 10000
Nb = 1000

strategy = QuasiRandomTraining(Nf+Nb; bcs_points=Nb,
  sampling_alg=SobolSample(), resampling=true)

discretization = PhysicsInformedNN(chain, strategy)

# @named pdesystem = PDESystem(eqs, bcs, domains, [x, y, z], [u(x, y, z), v(x, y, z), w(x, y, z), p(x, y, z)])
@named pdesystem = PDESystem(eqs, bcs, domains, [x, y, z], [u(x, y, z), v(x, y, z), w(x, y, z), p(x, y, z)])

prob = discretize(pdesystem, discretization)

sym_prob = symbolic_discretize(pdesystem, discretization)

pde_inner_loss_functions = sym_prob.loss_functions.pde_loss_functions
bcs_inner_loss_functions = sym_prob.loss_functions.bc_loss_functions

epoch = Base.RefValue{Int}(1)

callback = function (state, loss)
  println("epoch: ", epoch[])
  println("loss: ", loss)
  println("pde_losses: ", map(loss_ -> loss_(state), pde_inner_loss_functions))
  println("bcs_losses: ", map(loss_ -> loss_(state), bcs_inner_loss_functions))
  println("")
  epoch[] += 1
  return false
end

# res = Optimization.solve(prob, LBFGS(; linesearch = LineSearches.StrongWolfe()); callback = callback, maxiters = 10)

# phi = discretization.phi