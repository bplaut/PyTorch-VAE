import torch
from torch.utils.data import Sampler, DataLoader
import numpy as np

class ImgDifficultySampler(Sampler):
    """
    Sampler that tracks difficulty at the individual img level, rather than at the batch level.
    """
    def __init__(self, dataset_size, batch_size):
        raise NotImplementedError("This class needs to be modified because update_img_difficulties() now receives a list of image names instead of indices")
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epoch = 0        
        self.img_losses = np.ones(dataset_size) # Initialize equal weights for all imgs
            
    def __iter__(self):
        probs = self.img_losses / np.sum(self.img_losses)
        indices = np.random.choice(self.dataset_size, size=self.dataset_size, replace=True, p=probs)
        self.indices = indices

        return iter(indices.tolist())
    
    def __len__(self):
        return self.dataset_size
    
    def update_img_difficulties(self, indices, losses):
        print("\nNum of unique imgs sampled this epoch:  ", len(set(indices)))
        self.epoch += 1

        num_epochs_until_reset = 15
        if self.epoch % num_epochs_until_reset == 0:
            # reset all weights to be equal to the average loss from last epoch
            self.img_losses = np.ones(self.dataset_size) * np.mean(losses)
            print(f"{num_epochs_until_reset} epochs have passed, resetting image weights\n")
        else:
            img_counts = np.zeros(self.dataset_size, dtype=np.int32)
            # Update running average of loss for each img. Note that imgs not appearing in indices/losses (i.e., imgs which didn't get sampled this epoch) will retain their previous loss
            for idx, loss in zip(indices, losses):
                img_counts[idx] += 1
                self.img_losses[idx] = (self.img_losses[idx] * (img_counts[idx] - 1) + loss) / img_counts[idx]
            # print image with highest loss (among those that were sampled)
            max_loss_idx = np.argmax(self.img_losses[img_counts > 0])
            print("New img weights:  min={:.4f}, max={:.4f}, median={:.4f}, mean={:.4f}, std={:.4f}\n".format(np.min(1000 * self.img_losses), np.max(1000 * self.img_losses), np.median(1000 * self.img_losses), np.mean(1000 * self.img_losses), np.std(1000 * self.img_losses)))

