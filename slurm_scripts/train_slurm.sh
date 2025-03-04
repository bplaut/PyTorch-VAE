#!/bin/bash
#SBATCH --job-name=vae
#SBATCH --gpus=A6000:1
#SBATCH --time=3-00:00:00
#SBATCH --output=output_from_slurm/%j.out

cd /nas/ucb/bplaut/PyTorch-VAE
eval "$(/nas/ucb/bplaut/miniconda3/bin/conda shell.bash hook)"
conda activate /nas/ucb/bplaut/miniconda3/envs/vae

# Capture command-line inputs
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 train_dataset latent_dim kl_penalty config"
	exit 1
fi
train_dataset=$1
latent_dim=$2
kl_penalty=$3
config=$4

srun python run.py --train_dataset=$1 --latent_dim=$2 --kl_penalty=$3 --config=$4
