#!/bin/bash

source ~/georgito_env/bin/activate

module load gcc/9.4.0-eewq4j6
module load python/3.9.10-a7dicda

python training.py --model_name "v1" --num_hidden_layers 3 --hidden_layer_size 1000 --epochs 3000 --checkpoint_epochs 3000 --in_velocity 1 --rho 1 --mu 1 --x_max 1 --y_max 1 --z_max 1 --Nf 1000 --Nb 100 --Nw 1000 --Nu 2