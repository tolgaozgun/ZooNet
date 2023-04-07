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
import os

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

train_dir = "./data/train"
val_dir = "./data/val"

RANDOM_TRIALS = 1

def hyperparam_randomizer():
    lr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    epoch = [50, 100]
    batch_size = [16, 32, 64, 128]

    lr_i = randint(0, 4)
    epoch_i = randint(0, 1)
    batch_size_i = randint(0, 3)
    print(f"lr: {lr[lr_i]}")
    return lr[lr_i], batch_size[batch_size_i], epoch[epoch_i]

def normalize_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])



def create_run_dir(output_dir):
    # If output_dir is None, set it to the current directory
    if output_dir is None:
        output_dir = os.getcwd()

    # Get the list of directories in the folder that has the syntax "run_***" where *** is a number
    # Get the latest run number
    # Calculate the current run number by incrementing the latest run number by 1
    # Create a directory with the name "run_***" where *** is the current run number
    # Return the current run number
    print("Calculating run number...")

    run_number = 0
    
    for file in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, file)):
            if file.startswith("run_"):
                # Compare the current run number with the run number in the file name
                # If the run number in the file name is greater than the current run number, set the current run number to the run number in the file name
                if int(file[4:]) > run_number:
                    run_number = int(file[4:])

    run_number += 1

    # Set the directory name to "run_***" where *** is the current run number in 3 digits
    main_dir = os.path.join(output_dir, f"run_{run_number:04d}")
    os.mkdir(main_dir)
    
    # If subfolder is not None, create a subfolder in the run directory
    print(f"Run number: {run_number:04d}")
    return run_number, main_dir

def main():
    print("Loading settings...")
    settings = Settings()
    settings.load_from_file()

    run_dir = create_run_dir(settings.output_dir)
    
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
    # train_data = torch.utils.data.Subset(train_data, numpy.random.choice(len(train_data), 100_000, replace=False))
    val_data = ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_transform()
        ])
    )
    # val_data = torch.utils.data.Subset(val_data, numpy.random.choice(len(val_data), 1_000, replace=False))

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

        training_history, duration = train(train_data_loader, val_data_loader, train_steps, val_steps, classes=63, learning_rate=lr, epochs=epochs)

        # Write to excel
        for key in training_history.keys():
            if isinstance(training_history[key][0], torch.Tensor):
                training_history[key] = [t.detach().numpy() for t in training_history[key]]

        training_history["lr"] = lr
        training_history["batch_size"] = batch_size
        training_history["epochs"] = epochs
        df = pd.DataFrame.from_dict(training_history)
        df.to_excel(f"./{run_dir}/run{i}_history.xlsx")

        with open(f'./{run_dir}/run{i}.txt', mode='w') as f:
            f.write(f"Learning rate: {lr}. Number of epochs: {epochs}. Batch size: {batch_size}. Total duration: {duration} (s).\n")



main()
