import logging
from random import randint
import copy
import os
import argparse

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import numpy
import pandas as pd
from config import Config

from train import train
from dataset import load_dataset


# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

parser = argparse.ArgumentParser(
    prog="ZooNet",
    description="Command line interface for ZooNet.",
    epilog=""
)

parser.add_argument("-r", "--random", help="Randomly creates hyperparameters for the current run." + 
                    " If this option is selected any other flags providing a value to hyperparameters will be ignored",
                    type=int, default=None)
parser.add_argument("-l", "--learning-rate", help="Learning rate for the current epoch.", type=float, default=LEARNING_RATE)
parser.add_argument("-b", "--batch-size", help="Batch size for the current epoch.", type=int, default=BATCH_SIZE)
parser.add_argument("-e", "--epochs", help="Number of epochs for this run.", type=int, default=EPOCHS)


def hyperparam_randomizer():
    lr = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    epoch = [50, 100]
    batch_size = [16, 32, 64, 128]

    lr_i = randint(0, len(lr) - 1)
    epoch_i = randint(0, len(epoch) - 1)
    batch_size_i = randint(0, len(batch_size) - 1)

    return lr[lr_i], batch_size[batch_size_i], epoch[epoch_i]


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
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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


def training_history_to_excel(training_history: dict, out_path: str):
    df = pd.DataFrame.from_dict(training_history)
    df.to_excel(out_path)


def run_epoch(train_data, val_data, lr, batch_size, epochs, total_classes, i=None):
    print(f"Learning rate: {lr}, Batch size: {batch_size}, Total epochs: {epochs}")
    print("Initializing data loaders...")
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) 
    val_data_loader = DataLoader(val_data, batch_size=batch_size) 

    # Calculate steps per epoch
    train_steps = len(train_data_loader.dataset) // batch_size
    val_steps = len(val_data_loader.dataset) // batch_size
    print(f"Size of train set: {len(train_data_loader.dataset)}")
    print(f"Size of validation set: {len(val_data_loader.dataset)}")

    training_history, duration = train(train_data_loader, val_data_loader, train_steps, val_steps, 
                                        classes=int(total_classes), learning_rate=lr, epochs=epochs)

    # Converting these to normal CPU values is important when writing to excel
    for key in training_history.keys():
        if isinstance(training_history[key][0], torch.Tensor):
            training_history[key] = [t.cpu().detach().numpy() for t in training_history[key]]

    training_history["lr"] = lr
    training_history["batch_size"] = batch_size
    training_history["epochs"] = epochs

    if i == None:
        out_path = f"./{run_dir}/run1_history"
    else:
        out_path = f"./{run_dir}/run{i}_history"

    training_history_to_excel(training_history, f"{out_path}.xlsx")
    with open(f'{out_path}.txt', mode='w') as f:
        f.write(f"Learning rate: {lr}. Number of epochs: {epochs}. Batch size: {batch_size}. Total duration: {duration} (s).\n")


def main():
    args = parser.parse_args()
    random_trials = args.random
    lr = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs

    print("Loading settings...")
    config = Config()
    config.load_from_file()
    run_dir = create_run_dir(config.output_dir)
    train_dir = config.train_data_dir
    val_dir = config.val_data_dir
    total_classes = config.data_classes

    # Load the dataset, and split the dataset to train, validation, and test
    print("Loading the dataset...")
    train_data, val_data = load_dataset(train_dir, val_dir, train_subset_size=10000, val_subset_size=1000)

    if random_trials is not None:
        # Initialize  data loaders
        # Train the model
        for i in range(random_trials):
            lr, batch_size, epochs = hyperparam_randomizer()
            run_epoch(train_data, val_data, lr, batch_size, epochs, total_classes, i=i)
    else:
        run_epoch(train_data, val_data, lr, batch_size, epochs, total_classes)


main()
