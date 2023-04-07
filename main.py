import logging
import torch
import numpy
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from settings import Settings
from train import train
from random import randint
import pandas as pd
import copy

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

train_dir = "./data/train"
val_dir = "./data/val"

RANDOM_TRIALS = 1

def hyperparam_randomizer():
    lr = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1]
    epoch = [50, 100]
    batch_size = [16, 32, 64, 128]

    lr_i = randint(0, 4)
    epoch_i = randint(0, 1)
    batch_size_i = randint(0, 3)

    return lr[lr_i], batch_size[batch_size_i], epoch[epoch_i]

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
    # Train the model
    for i in range(RANDOM_TRIALS):
        print("Initializing data loaders...")
        lr, batch_size, epochs = hyperparam_randomizer()

        train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) 
        val_data_loader = DataLoader(val_data, batch_size=batch_size) 

        # Calculate steps per epoch
        train_steps = len(train_data_loader.dataset) // batch_size
        val_steps = len(val_data_loader.dataset) // batch_size
        print(f"Size of train set: {len(train_data_loader.dataset)}")
        print(f"Size of validation set: {len(val_data_loader.dataset)}")

        training_history, duration = train(train_data_loader, val_data_loader, train_steps, val_steps, classes=54, learning_rate=lr, epochs=epochs)

        # Write to excel
        for key in training_history.keys():
            if isinstance(training_history[key][0], torch.Tensor):
                training_history[key] = [t.detach().numpy() for t in training_history[key]]

        training_history["lr"] = lr
        training_history["batch_size"] = batch_size
        training_history["epochs"] = epochs
        df = pd.DataFrame.from_dict(training_history)
        df.to_excel(f"./result/run{i}_history.xlsx")

        with open(f'./result/run{i}.txt', mode='w') as f:
            f.write(f"Learning rate: {lr}. Number of epochs: {epochs}. Batch size: {batch_size}. Total duration: {duration} (s).\n")



main()
