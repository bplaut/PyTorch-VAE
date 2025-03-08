#!/bin/bash
#SBATCH --job-name=ae
#SBATCH --time=7-00:00:00
#SBATCH --output=output_from_slurm/%j.out
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --nodelist=ddpg.ist.berkeley.edu
#SBATCH --qos=high

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
