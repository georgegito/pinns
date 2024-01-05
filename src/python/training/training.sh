#!/bin/bash
#SBATCH --job-name=pinn_training
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --output=./out/output-%j.txt      ## use full path
#SBATCH --error=./out/error-%j.txt        ## use full path

source ~/georgito_env/bin/activate

module load gcc/9.4.0-eewq4j6
module load python/3.9.10-a7dicda

python training.py