#!/usr/bin/env python
# coding: utf-8

# # Physics Informed Neural Networks <br> F1 Car Front Wing Aerodymanics

# ## PINN

import os
from dotenv import load_dotenv
import sys

load_dotenv()

lib_dir = os.environ.get("LIB_DIR")
sys.path.append(lib_dir)

data_dir = os.environ.get("DATA_DIR")
model_dir = os.path.join(data_dir, "models")

points_filename = "points_final.csv"
measurements_filename = "measurements.csv"

import pandas as pd
import torch
import yaml
from pinn import PINN
import utils

device = utils.get_device()

print(f"Device: {device}")

wing_df = pd.read_csv(os.path.join(data_dir, points_filename))
measurements_df = pd.read_csv(os.path.join(data_dir, measurements_filename))

with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Air Density (rho): 1.2041kg/m^3
# Air Dynamic viscosity (mu): 1.81e-5 kg/m.s
rho = config["rho"]
mu = config["mu"]

# m/s
in_velocity = config["in_velocity"]

# Domain limits
x_max = config["x_max"]
y_max = config["y_max"]
z_max = config["z_max"]

input_dim = 3
output_dim = 4
hidden_units = config["hidden_units"]

model_name = config["model_name"]

pinn = PINN(input_dim, output_dim, hidden_units, model_name).to(device)

optimizer = torch.optim.LBFGS(pinn.parameters(), lr=1, line_search_fn="strong_wolfe")

epochs = config["epochs"]
checkpoint_epochs = config["checkpoint_epochs"]

Nf = config["Nf"]   # num of collocation points -> pde evaluation
Nb = config["Nb"]   # num of points to evaluate boundary conditions
Nw = config["Nw"]   # num of points of the surface of the front wing to evaluate boundary conditions
Nu = config["Nu"]   # num of points of real data


pinn.train_pinn(
  epochs=epochs, 
  optimizer=optimizer, 
  s_df=wing_df, 
  u_df=measurements_df, 
  Nf=Nf, 
  Nb=Nb, 
  Ns=Nw, 
  Nu=Nu, 
  x_max=x_max, 
  y_max=y_max, 
  z_max=z_max, 
  in_velocity=in_velocity, 
  mu=mu, 
  rho=rho, 
  device=device, 
  checkpoint_epochs=checkpoint_epochs, 
  model_dir=model_dir,
  c2=100, c3=100, c4=100, c5=100, c6=100, c8=100, c9=100
)