import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import zipfile
from difficulty_data_sampler import DifficultyDataSampler


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None, split='train', train_ratio=0.9):
        self.transform = transform
        self.split = split
        
        # Adjust this to match your dataset structure
        all_images = sorted([f for f in Path(data_dir).iterdir() if f.suffix in ['.jpg', '.png']], key=lambda x: int(x.stem))
        
        # Split into train/test sets (adjust ratio as needed)
        if split == 'train':
            self.images = all_images[:int(len(all_images) * train_ratio)]
        else:
            self.images = all_images[int(len(all_images) * train_ratio):]
        print(f"Loaded {len(self.images)} images from {data_dir} for {split} split.")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = default_loader(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, 0.0  # Return image and a dummy label
    
class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        train_dataset = 'coinrun',
        test_dataset = None,
        use_difficulty_sampling = False,
        softmax = False,
        **kwargs,
    ):
        super().__init__()

        self.train_data_dir = os.path.join(data_path, train_dataset) if train_dataset is not None else None
        self.test_data_dir = os.path.join(data_path, test_dataset) if test_dataset is not None else None
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_dataset_name = train_dataset
        self.test_dataset_name = test_dataset
        self.use_difficulty_sampling = use_difficulty_sampling
        self.batch_losses = {}
        self.softmax = softmax # only used for difficulty sampling

    def setup(self, stage: Optional[str] = None) -> None:    
        transform = transforms.Compose([transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])

        self.difficulty_sampler = None
        if self.train_dataset_name is not None:
            self.train_dataset = MyDataset(self.train_data_dir, split='train', transform=transform)
            self.val_dataset = MyDataset(self.train_data_dir, split='test', transform=transform)
            if self.use_difficulty_sampling:
                self.difficulty_sampler = DifficultyDataSampler(len(self.train_dataset), self.train_batch_size, softmax=self.softmax)
        else:
            self.train_dataset, self.val_dataset = None, None
        
        # If a separate test dataset is provided, use it; otherwise, use validation set
        if self.test_dataset_name is not None:
            self.test_dataset = MyDataset(self.test_data_dir, split='test', transform=transform, train_ratio=0)
        else:
            self.test_dataset = self.val_dataset
        
    def train_dataloader(self) -> DataLoader:
        if self.difficulty_sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                sampler=self.difficulty_sampler
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
            )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # Don't shuffle for test to get consistent results
            pin_memory=self.pin_memory,
        )

    def update_batch_difficulty(self, batch_idx, loss):
        self.batch_losses[batch_idx] = loss
    
    def on_epoch_end(self):
        if self.difficulty_sampler is not None:
            self.difficulty_sampler.update_weights(self.batch_losses)
        self.batch_losses = {}
