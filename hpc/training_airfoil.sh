#!/bin/bash

source ~/georgito_env/bin/activate

module load gcc/9.4.0-eewq4j6
module load python/3.9.10-a7dicda

num_hidden_layers=5
hidden_layer_size=50

# num_hidden_layers=10
# hidden_layer_size=50

# num_hidden_layers=20
# hidden_layer_size=50


# num_hidden_layers=5
# hidden_layer_size=100

# num_hidden_layers=10
# hidden_layer_size=100

# num_hidden_layers=20
# hidden_layer_size=100


# num_hidden_layers=5
# hidden_layer_size=200

# num_hidden_layers=10
# hidden_layer_size=200

# num_hidden_layers=20
# hidden_layer_size=200


# num_hidden_layers=2
# hidden_layer_size=500

# num_hidden_layers=3
# hidden_layer_size=500

# num_hidden_layers=5
# hidden_layer_size=500


epochs=1000
checkpoint_epochs=500

in_velocity=1
rho=1.
mu=0.01

x_min=-0.5
y_min=-0.5
x_max=1.5
y_max=0.5

Nf=10000
Nb=1000
Ns=1000
Nin=1000


python training_airfoil.py --num_hidden_layers $num_hidden_layers --hidden_layer_size $hidden_layer_size --epochs $epochs --checkpoint_epochs $checkpoint_epochs --in_velocity $in_velocity --rho $rho -mu $mu --x_min $x_min --y_min $y_min --x_max $x_max --y_max $y_max --Nf $Nf --Nb $Nb --Ns $Ns --Nin $Nin