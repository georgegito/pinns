# Aerodynamics of a Racing Car: 3D Velocity & Pressure Simulation using Physics-Informed Neural Networks (PINNs)

## Summary
An exploration aimed at using Physics-Informed Neural Networks (PINNs) to predict the aerodynamic characteristics of a racing car's front and rear wings. This research is centered on developing and implementing a PINN-based framework for simulating and predicting airflow velocity and pressure distribution over the wings' surfaces at steady state in a three-dimensional domain. The focus is on leveraging the advanced capabilities of PINNs to accurately forecast aerodynamic behaviors, enhancing the understanding and design of racing car aerodynamics.

## Implementation
The developed models feature a neural network architecture with three hidden layers, each consisting of 500 neurons. The network accepts three-dimensional spatial coordinates (x, y, z) as inputs, resulting in a three-component input dimension. The output dimension is four, corresponding to the three-dimensional velocity components (u, v, w) and pressure (p).

Each hidden layer is designed as a linear, fully-connected layer, utilizing the hyperbolic tangent (tanh) as the activation function. To initialize the model parameters, the Normal Xavier Initialization method is employed, which is known for its effectiveness in maintaining a balanced variance of activations and gradients throughout the network. The training of the model is conducted using the LBFGS optimizer, a quasi-Newton method known for its robustness and efficiency in handling small to moderately-sized models.

The loss function of the network is comprehensive and includes 10 distinct components to ensure accurate predictions:

1. **Navier-Stokes Equation**: Enforcing both the continuity and momentum aspects to simulate fluid dynamics accurately. 
2. **Poisson Equation**: To address pressure-velocity coupling, essential in fluid flow simulations.
3. **Inlet Velocity Boundary Condition (BC)**: Ensuring the model adheres to the predefined inlet velocity profiles.
4. **Outlet Pressure BC**: Setting the pressure conditions at the outlet, crucial for flow directionality and stability.
5. **Left Slip BC**: Implementing slip boundary conditions on the left boundary, allowing fluid to slide along the surface without penetration.
6. **Right Slip BC**: Similar to the left slip, but applied on the right boundary.
7. **Up Slip BC**: Applying slip conditions on the upper boundary of the domain.
8. **Down BC**: Differentiating between no-slip conditions for the front wing and slip conditions for the rear, to capture the distinct aerodynamic behaviors of each.
9. **No-Slip Surface BC**: Ensuring that the fluid adheres to the solid surfaces, a key aspect in aerodynamic simulations.
10. **Real Data**: Integrating real-world data into the model to enhance its predictive accuracy and validity.

This elaborate loss function ensures that the neural network not only learns the underlying physics of fluid dynamics but also adheres to the specific boundary conditions and real-world data, making it a capable tool for predicting the aerodynamic characteristics of racing car wings.

**TODO**: add loss function formula

**TODO**: describe ReLoBRaLo algorithm for adaptive loss

## Results
### Front Wing Simulation - AtmosElegantNavigator105
#### Velocity
![fig1](https://github.com/georgegito/pinns/blob/main/data/front_wing/fig/velocity1.png)
![fig1](https://github.com/georgegito/pinns/blob/main/data/front_wing/fig/velocity2.png)

#### Pressure
![fig1](https://github.com/georgegito/pinns/blob/main/data/front_wing/fig/pressure1.png)
![fig1](https://github.com/georgegito/pinns/blob/main/data/front_wing/fig/pressure2.png)

#### Learning Curves
![fig1](https://github.com/georgegito/pinns/blob/main/data/front_wing/fig/learning_curves.png)

#### Adaptive Loss Function Parameters - Lambdas
![fig1](https://github.com/georgegito/pinns/blob/main/data/front_wing/fig/lambdas.png)

### Rear Wing Simulation - VortexPreciseInnovator894
#### Velocity
![fig1](https://github.com/georgegito/pinns/blob/main/data/rear_wing/fig/velocity1.png)
![fig1](https://github.com/georgegito/pinns/blob/main/data/rear_wing/fig/velocity2.png)

#### Pressure
TODO

#### Learning Curves
![fig1](https://github.com/georgegito/pinns/blob/main/data/rear_wing/fig/learning_curves.png)

#### Adaptive Loss Function Parameters - Lambdas
![fig1](https://github.com/georgegito/pinns/blob/main/data/rear_wing/fig/lambdas.png)
