#!/bin/bash

# Function to print usage information
usage() {
    echo "Usage: $0 train_dataset \"latent_dim1 latent_dim2 ...\" \"kl_penalty1 kl_penalty2 ...\" config"
    echo "Example: $0 coinrun \"128 256 512\" \"0.1 0.01 0.001\" configs/vae.yaml"
    exit 1
}

# Check if we have the right number of arguments
if [ "$#" -ne 4 ]; then
    usage
fi

# Parse command-line arguments
train_dataset=$1
latent_dims=($2)  # Convert space-separated string to array
kl_penalties=($3) # Convert space-separated string to array
config=$4

# Validate inputs
if [ ${#latent_dims[@]} -eq 0 ]; then
    echo "Error: No latent dimensions provided"
    usage
fi

if [ ${#kl_penalties[@]} -eq 0 ]; then
    echo "Error: No KL penalties provided"
    usage
fi

# Print configuration
echo "============================================"
echo "Batch VAE Training Configuration"
echo "============================================"
echo "Train Dataset: $train_dataset"
echo "Latent Dimensions: ${latent_dims[@]}"
echo "KL Penalties: ${kl_penalties[@]}"
echo "Config File: $config"
echo "Total Jobs to Submit: $((${#latent_dims[@]} * ${#kl_penalties[@]}))"
echo "============================================"
echo

# Confirm with user
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled by user"
    exit 0
fi

# Submit jobs for all combinations
for latent_dim in "${latent_dims[@]}"; do
    for kl_penalty in "${kl_penalties[@]}"; do
        echo "Submitting job: train_dataset=$train_dataset, latent_dim=$latent_dim, kl_penalty=$kl_penalty"
        
        # Submit the job using sbatch and capture the job ID
        job_id=$(sbatch --parsable train_slurm.sh "$train_dataset" "$latent_dim" "$kl_penalty" "$config")
        
        if [ $? -eq 0 ]; then
            echo "Successfully submitted job $job_id"
        else
            echo "Failed to submit job"
        fi
    done
done
