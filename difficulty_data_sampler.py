import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np

class DifficultyDataSampler(Sampler):
    """
    Sampler that prioritizes difficult examples based on their reconstruction loss.
    """
    def __init__(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        # We're only going to sample from full batches (floor), but we'll use the ceiling to determine the total samples to align with what PyTorch Lightning expects
        self.num_batches_available = dataset_size // batch_size
        self.num_batches_to_sample = (dataset_size + batch_size - 1) // batch_size
        self.batch_weights = np.ones(self.num_batches_available)
        self.batch_weights = self.batch_weights / np.sum(self.batch_weights)
        
    def __iter__(self):
        # Sample batch indices based on difficulty weights
        batch_indices = np.random.choice(
            self.num_batches_available, 
            size=self.num_batches_to_sample, 
            replace=True,
            p=self.batch_weights)
        
        # Convert batch indices to sample indices
        indices = []
        for batch_idx in batch_indices:
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.dataset_size)
            batch_indices = list(range(start_idx, end_idx))
            indices.extend(batch_indices)
            
        return iter(indices)
    
    def __len__(self):
        return self.dataset_size
    
    def update_weights(self, batch_losses):
        """
        Update batch weights based on their losses.
        
        Args:
            batch_losses: Dictionary mapping batch indices to their losses
        """
        # Ensure all batches have a weight
        for batch_idx in range(self.num_batches_available):
            if batch_idx not in batch_losses:
                batch_losses[batch_idx] = 0.0
                
        # Create new weights based on losses
        new_weights = np.array([batch_losses[i] for i in range(self.num_batches_available)])
        self.batch_weights = new_weights / np.sum(new_weights)
        # print weight stats to 4 decimal places, scaled by 1000 for readability
        print("\nNew batch weights:  min={:.4f}, max={:.4f}, median={:.4f}, mean={:.4f}, std={:.4f}\n".format(1000 * np.min(self.batch_weights), 1000 * np.max(self.batch_weights), 1000 * np.median(self.batch_weights),1000 * np.mean(self.batch_weights),1000 * np.std(self.batch_weights)))

