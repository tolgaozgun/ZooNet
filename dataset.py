import os

import numpy as np
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

mean=[0.4914, 0.4822, 0.4465]
std=[0.2023, 0.1994, 0.2010]

classes_path = os.path.join("./data/train")
classes = os.listdir(classes_path)
idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

def unnormalize_im(im):
    unnormalize_transform = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )
    return unnormalize_transform(im)

class CustomDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.total_imgs = []
        label_dirs = os.listdir(main_dir)
        for label_dir in label_dirs:
            labeled_img_dir = os.path.join(main_dir, label_dir)
            all_imgs = os.listdir(labeled_img_dir)
            all_img_paths = [os.path.join(labeled_img_dir, img_path) for img_path in all_imgs]
            self.total_imgs.extend(all_img_paths)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)

        label = img_loc.split('\\')[-2]
        label_id = class_to_idx[label]

        return tensor_image, label_id

def load_custom_dataset():
    pass


def load_dataset(train_dir: str, val_dir: str, train_crop: int=128, val_crop: int=128,
                train_subset_size: int=None, val_subset_size: int=None):
    print(f"Train data directory: {train_dir}. Validation data directory: {val_dir}")
    normalize_transform = transforms.Normalize(
        mean=mean,
        std=std,
    )
    train_data = ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize_transform
        ])
    )
    train_len = int(len(train_data) * 0.9)
    test_len = len(train_data) - train_len
    train_data, test_data = torch.utils.data.random_split(train_data, [train_len, test_len])

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

    return train_data, val_data, test_data
