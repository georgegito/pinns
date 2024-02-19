import os
from dotenv import load_dotenv
import sys
import yaml

# load environment variables
load_dotenv()

lib_dir = os.environ.get("HPC_LIB_DIR")
sys.path.append(lib_dir)

data_dir = os.environ.get("HPC_DATA_DIR_AIRFOIL")
model_dir = os.path.join(data_dir, "models")

import pandas as pd
import torch
from airfoil_pinn import AirfoilPINN
from naca4digit_airfoil import Naca4DigitAirfoil
import utils

device = utils.get_device()
print(f"Device: {device}")


with open("config_airfoil.yaml", 'r') as file:
    config = yaml.safe_load(file)


rho = config["rho"]
mu = config["mu"]

in_velocity = config["in_velocity"]
out_pressure = config["out_pressure"]


# Domain limits
domain1 = utils.Domain2D(
        x_min=config["domain1"]["x_min"], 
        x_max=config["domain1"]["x_max"], 
        y_min=config["domain1"]["y_min"],
        y_max=config["domain1"]["y_max"])

domain2 = utils.Domain2D(
        x_min=config["domain2"]["x_min"], 
        x_max=config["domain2"]["x_max"], 
        y_min=config["domain2"]["y_min"],
        y_max=config["domain2"]["y_max"])


# Define airfoil parameters (example: NACA 2412)
chord = 1.0
m = 0.02  # maximum camber
p = 0.4   # position of maximum camber
t = 0.12  # maximum thickness
num_points = 100
angle_of_attack = 10

airfoil = Naca4DigitAirfoil(chord, m, p, t, alpha_deg=angle_of_attack)

center_x, center_y = (domain1.x_max + domain1.x_min) / 2, (domain1.y_max + domain1.y_min) / 2

# Translate to the origin
airfoil.translate(-center_x, -center_y)
domain1.translate(-center_x, -center_y)
domain2.translate(-center_x, -center_y)

# Scale the airfoil and the domain
width = domain1.x_max - domain1.x_min
height = domain1.y_max - domain1.y_min
scaling_factor = 1 / max(width, height)

airfoil.scale(sx=scaling_factor, sy=scaling_factor)
domain1.scale(sx=scaling_factor, sy=scaling_factor)
domain2.scale(sx=scaling_factor, sy=scaling_factor)


num_hidden_layers = config["num_hidden_layers"]
hidden_layer_size = config["hidden_layer_size"]
hidden_layers = [hidden_layer_size for _ in range(num_hidden_layers)]

if config["activation_function"] == "tanh":
    activation_function = torch.tanh
elif config["activation_function"] == "sigmoid":
    activation_function = torch.sigmoid
else:
    raise ValueError("Activation function not recognized")


pinn = AirfoilPINN(hidden_layers, activation_function, airfoil).to(device)


if config["optimizer"]["name"] == "adam":
    optimizer = torch.optim.Adam(pinn.parameters(), lr=config["optimizer"]["lr"])
elif config["optimizer"]["name"] == "lbfgs":
    optimizer = torch.optim.LBFGS(pinn.parameters(), lr=config["optimizer"]["lr"], line_search_fn="strong_wolfe")
else:
    raise ValueError("Optimizer not recognized")


epochs = config["epochs"]
checkpoint_epochs = config["checkpoint_epochs"]

Nf1 = config["Nf1"]   # num of collocation points -> pde evaluation - domain 1
Nf2 = config["Nf2"]   # num of collocation points -> pde evaluation - domain 2
Nb  = config["Nb"]   # num of points to evaluate boundary conditions
Ns  = config["Ns"]   # num of points of the surface of the airfoil to evaluate boundary conditions
Nin = config["Nin"] # num of points inside the airfoil to evaluate boundary conditions


pinn.train_pinn(
  epochs=epochs, 
  optimizer=optimizer, 
  Nf1=Nf1, 
  Nf2=Nf2, 
  Nb=Nb, 
  Ns=Ns, 
  Nin=Nin,
  domain1=domain1,
  domain2=domain2,
  in_velocity=in_velocity, 
  out_pressure=out_pressure, 
  mu=mu, 
  rho=rho, 
  device=device, 
  checkpoint_epochs=checkpoint_epochs, 
  model_dir=model_dir, 
)