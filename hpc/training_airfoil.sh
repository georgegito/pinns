#!/bin/bash

source ~/georgito_env/bin/activate

module load gcc/9.4.0-eewq4j6
module load python/3.9.10-a7dicda

python training_airfoil.py --num_hidden_layers 3 --hidden_layer_size 500 --epochs 3000 --checkpoint_epochs 300 --in_velocity 1 --rho 1.225 --mu 0 --x_min -0.5 --y_min 0.5 --x_max 1.5 --y_max 0.5 --Nf 10000 --Nb 100 --Ns 10000