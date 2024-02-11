import os
from dotenv import load_dotenv
import sys
import argparse

# load environment variables
load_dotenv()

lib_dir = os.environ.get("HPC_LIB_DIR")
sys.path.append(lib_dir)

data_dir = os.environ.get("HPC_DATA_DIR_AIRFOIL")
model_dir = os.path.join(data_dir, "models")

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--num_hidden_layers', type=int, required=True)
parser.add_argument('--hidden_layer_size', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--checkpoint_epochs', type=int, required=True)
parser.add_argument('--in_velocity', type=float, required=True)
parser.add_argument('--rho', type=float, required=True)
parser.add_argument('--mu', type=float, required=True)
parser.add_argument('--x_min', type=float, required=True)
parser.add_argument('--x_max', type=float, required=True)
parser.add_argument('--y_min', type=float, required=True)
parser.add_argument('--y_max', type=float, required=True)
parser.add_argument('--Nf', type=int, required=True, help='num of collocation points')
parser.add_argument('--Nb', type=int, required=True, help='num of points to evaluate boundary conditions')
parser.add_argument('--Ns', type=int, required=True, help='num of points of the surface of the front wing to evaluate boundary conditions')

args = parser.parse_args()

rho = args.rho
mu = args.mu
in_velocity = args.in_velocity
x_min = args.x_min
x_max = args.x_max
y_min = args.y_min
y_max = args.y_max
num_hidden_layers = args.num_hidden_layers
hidden_layer_size = args.hidden_layer_size
epochs = args.epochs
checkpoint_epochs = args.checkpoint_epochs
Nf = args.Nf
Nb = args.Nb
Ns = args.Ns

import pandas as pd
import torch
from airfoil_pinn import AirfoilPINN
from naca4digit_airfoil import Naca4DigitAirfoil
import utils

device = utils.get_device()
print(f"Device: {device}")

chord = 1.0
m = 0.02
p = 0.4
t = 0.12
num_points = 100

airfoil = Naca4DigitAirfoil(chord, m, p, t)

hidden_layers = [hidden_layer_size for _ in range(num_hidden_layers)]

pinn = AirfoilPINN(hidden_layers, airfoil).to(device)

optimizer = torch.optim.LBFGS(pinn.parameters(), lr=1, line_search_fn="strong_wolfe")


pinn.train_pinn(
  epochs=epochs, 
  optimizer=optimizer, 
  Nf=Nf, 
  Nb=Nb, 
  Ns=Ns, 
  x_min=x_min, 
  x_max=x_max, 
  y_min=y_min, 
  y_max=y_max, 
  in_velocity=in_velocity, 
  mu=mu, 
  rho=rho, 
  device=device, 
  checkpoint_epochs=checkpoint_epochs, 
  model_dir=model_dir, 
)

pinn.plot_learning_curves(save=True, output_dir=model_dir)