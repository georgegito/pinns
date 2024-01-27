import os
from dotenv import load_dotenv
import sys
import argparse

# load environment variables
load_dotenv()

lib_dir = os.environ.get("HPC_LIB_DIR")
sys.path.append(lib_dir)

data_dir_front_wing = os.environ.get("HPC_DATA_DIR_FRONT_WING")
data_dir_rear_wing = os.environ.get("HPC_DATA_DIR_REAR_WING")

model_dir_front_wing = os.path.join(data_dir_front_wing, "models")
model_dir_rear_wing = os.path.join(data_dir_rear_wing, "models")

points_filename = "points_final.csv"
measurements_filename = "measurements.csv"

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--num_hidden_layers', type=int, required=True)
parser.add_argument('--hidden_layer_size', type=int, required=True)
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
parser.add_argument('--object', type=str, required=True)

args = parser.parse_args()

rho = args.rho
mu = args.mu
in_velocity = args.in_velocity
x_max = args.x_max
y_max = args.y_max
z_max = args.z_max
num_hidden_layers = args.num_hidden_layers
hidden_layer_size = args.hidden_layer_size
epochs = args.epochs
checkpoint_epochs = args.checkpoint_epochs
Nf = args.Nf
Nb = args.Nb
Nw = args.Nw
Nu = args.Nu

if args.object == "FRONT_WING":
    data_dir = data_dir_front_wing
    model_dir = model_dir_front_wing
elif args.object == "REAR_WING":
    data_dir = data_dir_rear_wing
    model_dir = model_dir_rear_wing
else:
    raise ValueError("Invalid object")

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
hidden_layers = [hidden_layer_size for _ in range(num_hidden_layers)]

pinn = PINN(input_dim, output_dim, hidden_layers).to(device)

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