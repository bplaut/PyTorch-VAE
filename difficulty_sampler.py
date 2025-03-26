import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np
import os

class ImgDifficultySampler(Sampler):
    """
    Sampler that tracks difficulty at the individual img level, rather than at the batch level.
    """
    def __init__(self, dataset, batch_size):
        self.dataset_size = len(dataset)
        self.batch_size = batch_size
        self.img_weights = np.ones(self.dataset_size) # Initialize all images to have equal weight
        self.image_paths = dataset.images
        self.imgname_to_idx = {os.path.basename(path): idx for idx, path in enumerate(self.image_paths)}
            
    def __iter__(self):
        probs = self.img_weights / np.sum(self.img_weights)
        indices = np.random.choice(self.dataset_size, size=self.dataset_size, replace=True, p=probs)
        self.indices = indices

        return iter(indices.tolist())
    
    def __len__(self):
        return self.dataset_size
    
    def update_img_difficulties(self, img_names, losses):
        # Convert filenames to indices
        indices = [self.imgname_to_idx[img_name] for img_name in img_names]
        
        img_counts = np.zeros(self.dataset_size, dtype=np.int32)
        # Update running average of loss for each img. Images not appearing in indices/losses (i.e., imgs which didn't get sampled this epoch) will retain their previous loss
        for idx, loss in zip(indices, losses):
            img_counts[idx] += 1
            self.img_weights[idx] = (self.img_weights[idx] * (img_counts[idx] - 1) + loss) / img_counts[idx]
        weight_floor = np.mean(self.img_weights) / 5 # don't let weights get too small so we don't ignore any images
        self.img_weights = np.maximum(self.img_weights, weight_floor)
        min_expected_samples = len(indices) * np.min(self.img_weights) / np.sum(self.img_weights)
        print("\nNum of unique imgs sampled this epoch:  ", len(set(indices)))
        print("New img weights:  min={:.4f}, max={:.4f}, median={:.4f}, mean={:.4f}, floor={:.4f}, min expected samples={:.4f}\n".format(
            np.min(1000 * self.img_weights), 
            np.max(1000 * self.img_weights), 
            np.median(1000 * self.img_weights), 
            np.mean(1000 * self.img_weights), 
            1000 * weight_floor,
            min_expected_samples))
