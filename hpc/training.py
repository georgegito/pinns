import os
from dotenv import load_dotenv
import sys
import argparse

# load environment variables
load_dotenv()

lib_dir = os.environ.get("LIB_DIR")
sys.path.append(lib_dir)

data_dir = os.environ.get("DATA_DIR")
model_dir = os.path.join(data_dir, "models")

points_filename = "points_final.csv"
measurements_filename = "measurements.csv"

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--hidden_units', nargs=3, type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--checkpoint_epochs', type=int, required=True)
parser.add_argument('--in_velocity', type=float, required=True)
parser.add_argument('--rho', type=float, required=True)
parser.add_argument('--mu', type=float, required=True)
parser.add_argument('--x_max', type=float, required=True)
parser.add_argument('--y_max', type=float, required=True)
parser.add_argument('--z_max', type=float, required=True)
parser.add_argument('--Nf', type=int, required=True, help='num of collocation points')
parser.add_argument('--Nb', type=int, required=True, help='num of points to evaluate boundary conditions')
parser.add_argument('--Nw', type=int, required=True, help='num of points of the surface of the front wing to evaluate boundary conditions')
parser.add_argument('--Nu', type=int, required=True, help='num of points of real data')

args = parser.parse_args()

rho = args.rho
mu = args.mu
in_velocity = args.in_velocity
x_max = args.x_max
y_max = args.y_max
z_max = args.z_max
hidden_units = args.hidden_units
model_name = args.model_name
epochs = args.epochs
checkpoint_epochs = args.checkpoint_epochs
Nf = args.Nf
Nb = args.Nb
Nw = args.Nw
Nu = args.Nu


import pandas as pd
import torch
from pinn import PINN
import utils

device = utils.get_device()
print(f"Device: {device}")

wing_df = pd.read_csv(os.path.join(data_dir, points_filename))
measurements_df = pd.read_csv(os.path.join(data_dir, measurements_filename))

input_dim = 3
output_dim = 4

pinn = PINN(input_dim, output_dim, hidden_units, model_name).to(device)

optimizer = torch.optim.LBFGS(pinn.parameters(), lr=1, line_search_fn="strong_wolfe")


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
)