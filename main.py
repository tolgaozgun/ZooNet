import logging
import torch
import numpy
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from settings import Settings
from train import train

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

train_dir = "./data/train"
val_dir = "./data/val"

def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

def main():
    print("Loading settings...")
    settings = Settings()
    settings.load_from_file()
    
    # Load the dataset, and split the dataset to train, validation, and test
    print("Loading the dataset...")
    train_data = ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
        ])
    )
    train_data = torch.utils.data.Subset(train_data, numpy.random.choice(len(train_data), 1000, replace=False))
    val_data = ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
        ])
    )
    val_data = torch.utils.data.Subset(val_data, numpy.random.choice(len(val_data), 100, replace=False))

    # Initialize  data loaders
    print("Initializing data loaders...")
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE) 
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE) 

    # Calculate steps per epoch
    train_steps = len(train_data_loader.dataset) // BATCH_SIZE
    val_steps = len(val_data_loader.dataset) // BATCH_SIZE
    print(f"Size of train set: {len(train_data_loader.dataset)}")
    print(f"Size of validation set: {len(val_data_loader.dataset)}")

    # Train the model
    train(train_data_loader, val_data_loader, train_steps, val_steps, classes=11)


main()
