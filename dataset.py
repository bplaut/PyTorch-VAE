import os
import re
import torch
import random
from torch import Tensor
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import zipfile
from difficulty_sampler import ImgDifficultySampler

class MyDataset(Dataset):
    def __init__(self, images, transform=None):
        self.transform = transform
        self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = default_loader(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, 0.0, os.path.basename(img_path) # 0.0 is a dummy label


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
        self.sampled_img_names = []
        self.sampled_img_losses = []

    def setup(self, stage: Optional[str] = None) -> None:    
        transform = transforms.Compose([transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])

        self.difficulty_sampler = None
        if self.train_dataset_name is not None:
            (train_files, val_files) = self.split_images(self.train_data_dir, train_ratio=0.9)
            self.train_dataset = MyDataset(train_files, transform=transform)
            self.val_dataset = MyDataset(val_files, transform=transform)
            if self.use_difficulty_sampling:
                self.difficulty_sampler = ImgDifficultySampler(len(self.train_dataset), self.train_batch_size)
        else:
            self.train_dataset, self.val_dataset = None, None
        
        # If a separate test dataset is provided, use it; otherwise, use validation set
        if self.test_dataset_name is not None:
            (_, test_files) = self.split_images(self.test_data_dir, train_ratio=0)
            self.test_dataset = MyDataset(test_files, transform=transform)
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

    def record_img_losses(self, img_names, losses):
        if isinstance(img_names, torch.Tensor):
            img_names = img_names.cpu().tolist()
        if isinstance(losses, torch.Tensor):
            losses = losses.cpu().tolist()
        self.sampled_img_names.extend(img_names)
        self.sampled_img_losses.extend(losses)
    
    def on_epoch_end(self):
        if self.difficulty_sampler is not None:
            self.difficulty_sampler.update_img_difficulties(self.sampled_img_names, self.sampled_img_losses)
            self.sampled_img_names = []
            self.sampled_img_losses = []
            
    def split_images(self, data_dir, train_ratio):
        print(f"\nLoading images from {data_dir}")
        # If we have complex files names (with run-id, iter, env), we'll randomly select runs for train and test. Otherwise we'll just split the files.
        simple_pattern = re.compile(r'^(\d+)\.png$')
        complex_pattern = re.compile(r'iter(\d+).*?env(\d+).*?step(\d+).*?run-id(\d+)')
        # First collect the set of all run-ids
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        run_ids = set()
        for f in image_files:
            complex_match = complex_pattern.search(f)
            if complex_match:
                run_ids.add(int(complex_match.group(4)))
        run_ids = list(run_ids)
        random.shuffle(run_ids)
        train_run_ids = run_ids[:int(train_ratio * len(run_ids))]
        test_run_ids = run_ids[int(train_ratio * len(run_ids)):]
        print(f"Found {len(run_ids)} runs of data generation, splitting into {len(train_run_ids)} training run-ids and {len(test_run_ids)} test run-ids")

        simple_filepaths = []
        train_complex_filepaths = []
        test_complex_filepaths = []
        for f in image_files:
            simple_match = simple_pattern.match(f)
            complex_match = complex_pattern.search(f)
            if simple_match:
                simple_filepaths.append(os.path.join(data_dir, f))
            elif complex_match:
                run_id = int(complex_match.group(4))
                if run_id in train_run_ids:
                    train_complex_filepaths.append(os.path.join(data_dir, f))
                else:
                    test_complex_filepaths.append(os.path.join(data_dir, f))
        random.shuffle(simple_filepaths)
        train_simple_filepaths = simple_filepaths[:int(train_ratio * len(simple_filepaths))]
        test_simple_filepaths = simple_filepaths[int(train_ratio * len(simple_filepaths)):]
        train_all = train_simple_filepaths + train_complex_filepaths
        test_all = test_simple_filepaths + test_complex_filepaths
        test_all = self.sort_images(test_all) # Keep test images in order so we can make gifs and such
        print(f"Loaded {len(train_all)} training images and {len(test_all)} test images")
        return (train_all, test_all)
    
    def sort_images(self, img_paths):
        """
        Sort image files in a directory:
        - Simple filenames (e.g., '5.png') are sorted by their integer value
        - Complex filenames sorted by run-id, then iter, then env, then step
        Theoretically you should never have both simple and complex filenames in the same directory but we handle that just in case.
        """
        # Regex pattern for simple filenames (e.g., "5.png")
        simple_pattern = re.compile(r'^(\d+)\.png$')

        # Regex pattern for complex filenames with named groups
        complex_pattern = re.compile(r'iter(\d+).*?env(\d+).*?step(\d+).*?run-id(\d+)')

        def get_sort_key(filepath):
            filename = os.path.basename(filepath)
            simple_match = simple_pattern.match(filename)
            complex_match = complex_pattern.search(filename)
            if simple_match:
                return (0, int(simple_match.group(1)))
            elif complex_match:
                run_id = int(complex_match.group(4))
                iter_num = int(complex_match.group(1))
                env_num = int(complex_match.group(2))
                step_num = int(complex_match.group(3))
                return (1, run_id, iter_num, env_num, step_num)
            else:
                raise ValueError(f"Filename {filename} is in the wrong format")
        return sorted(img_paths, key=get_sort_key)

