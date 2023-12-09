using NeuralPDE
using Serialization
using OptimizationOptimJL

prob = deserialize("/Users/ggito/repos/pinns/src/julia/models/pinn")

epoch = 0

callback = function (p, l)
    global epoch
    epoch += 1
    # if epoch % 10 == 0
        println("Epoch: $epoch\tLoss: $l")
    # end
    return false
end

opt = Optim.BFGS()
# opt = Optim.GradientDescent(P=0.01)

println("Training PINN")

display(prob)

res = @time Optimization.solve(prob, opt; callback=callback, maxiters=150)