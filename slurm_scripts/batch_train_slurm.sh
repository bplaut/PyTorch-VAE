#!/bin/bash

# Function to print usage information
usage() {
    echo "Usage: $0 train_dataset \"latent_dim1 latent_dim2 ...\" config"
    echo "Example: $0 coinrun \"128 256 512\" pure_ae.yaml"
    exit 1
}

# Check if we have the right number of arguments
if [ "$#" -ne 3 ]; then
    usage
fi

# Parse command-line arguments
train_dataset=$1
latent_dims=($2)  # Convert space-separated string to array
config=$3
config_path="configs/$config"

# Validate inputs
if [ ${#latent_dims[@]} -eq 0 ]; then
    echo "Error: No latent dimensions provided"
    usage
fi
# Print configuration
echo "============================================"
echo "Batch VAE Training Configuration"
echo "============================================"
echo "Train Dataset: $train_dataset"
echo "Latent Dimensions: ${latent_dims[@]}"
echo "Config File: $config"
echo "Total Jobs to Submit: $((${#latent_dims[@]}))"
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
    echo "Submitting job: train_dataset=$train_dataset, latent_dim=$latent_dim"
        
    exp_name="$config-$latent_dim-$train_dataset"
    # Submit the job using sbatch and capture the job ID
    job_id=$(sbatch --parsable --output=output_from_slurm/$exp_name.out train_slurm.sh "$train_dataset" "$latent_dim" "$config_path")
    
    if [ $? -eq 0 ]; then
        echo "Successfully submitted job $job_id"
    else
        echo "Failed to submit job"
    fi
done
