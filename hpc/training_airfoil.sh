#!/bin/bash

source ~/georgito_env/bin/activate

module load gcc/9.4.0-eewq4j6
module load python/3.9.10-a7dicda

python training_airfoil.py