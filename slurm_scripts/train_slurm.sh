#!/bin/bash
#SBATCH --job-name=ae
#SBATCH --time=7-00:00:00
#SBATCH --gpus=1
#SBATCH --qos=high
#SBATCH --mem=15gb

cd /nas/ucb/bplaut/PyTorch-VAE
eval "$(/nas/ucb/bplaut/miniconda3/bin/conda shell.bash hook)"
conda activate /nas/ucb/bplaut/miniconda3/envs/vae

# Capture command-line inputs
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 train_dataset latent_dim config"
	exit 1
fi
train_dataset=$1
latent_dim=$2
config=$3

srun python run.py --train_dataset=$1 --latent_dim=$2 --config=$3
