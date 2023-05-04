import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

def load_dataset(train_dir: str, val_dir: str, train_crop: int=128, val_crop: int=128,
                train_subset_size: int=None, val_subset_size: int=None):
    print(f"Train data directory: {train_dir}. Validation data directory: {val_dir}")
    normalize_transform = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    train_data = ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize_transform
        ])
    )
    if train_subset_size:
        train_data = torch.utils.data.Subset(train_data, np.random.choice(len(train_data), train_subset_size, replace=False))

    val_data = ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize_transform
        ])
    )
    if val_subset_size:
        val_data = torch.utils.data.Subset(val_data, np.random.choice(len(val_data), val_subset_size, replace=False))

    return train_data, val_data
