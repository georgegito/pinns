#!/bin/bash

source ~/georgito_env/bin/activate

module load gcc/9.4.0-eewq4j6
module load python/3.9.10-a7dicda

# v1
python training_airfoil.py --num_hidden_layers 3 --hidden_layer_size 500 --epochs 1000 --checkpoint_epochs 500 --in_velocity 1 --rho 1.225 --mu 0 --x_min -0.5 --y_min -0.5 --x_max 1.5 --y_max 0.5 --Nf 10000 --Nb 1000 --Ns 1000 --Nin 1000

# # v2
# python training_airfoil.py --num_hidden_layers 10 --hidden_layer_size 64 --epochs 1000 --checkpoint_epochs 500 --in_velocity 1 --rho 1.225 --mu 0 --x_min -0.5 --y_min -0.5 --x_max 1.5 --y_max 0.5 --Nf 10000 --Nb 1000 --Ns 1000 --Nin 1000

# # v3
# python training_airfoil.py --num_hidden_layers 5 --hidden_layer_size 100 --epochs 1000 --checkpoint_epochs 500 --in_velocity 1 --rho 1.225 --mu 0 --x_min -0.5 --y_min -0.5 --x_max 1.5 --y_max 0.5 --Nf 10000 --Nb 1000 --Ns 1000 --Nin 1000

# # v4
# python training_airfoil.py --num_hidden_layers 3 --hidden_layer_size 1000 --epochs 1000 --checkpoint_epochs 500 --in_velocity 1 --rho 1.225 --mu 0 --x_min -0.5 --y_min -0.5 --x_max 1.5 --y_max 0.5 --Nf 10000 --Nb 1000 --Ns 1000 --Nin 1000