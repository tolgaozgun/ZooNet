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
from utils import create_run_dir, hyperparam_randomizer, training_history_to_excel
from dataset import load_dataset


# Hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
MODEL_OUT_PATH = "./"

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
parser.add_argument("-s", "--save", help="Saves the learned model parameters to the given path.", type=str, default=MODEL_OUT_PATH)
parser.add_argument("--load-model", help="Loads the model parameters for running tests. No learning is made if this flag is true, only the test input is run.", type=str, default=None)
parser.add_argument("-t", "--transfer-learning", help="Enables transfer learning, runs the pretrained model on a different dataset.")

def run_epoch(train_data, val_data, lr, batch_size, epochs, total_classes, run_dir, i=None):
    print(f"Learning rate: {lr}, Batch size: {batch_size}, Total epochs: {epochs}")
    print("Initializing data loaders...")
    print(run_dir)
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) 
    val_data_loader = DataLoader(val_data, batch_size=batch_size) 

    # Calculate steps per epoch
    train_steps = len(train_data_loader.dataset) // batch_size
    val_steps = len(val_data_loader.dataset) // batch_size
    print(f"Size of train set: {len(train_data_loader.dataset)}")
    print(f"Size of validation set: {len(val_data_loader.dataset)}")

    training_history, duration, model = train(train_data_loader, val_data_loader, train_steps, val_steps, 
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
    # TODO: Draw plots

    return model


def run_test():
    pass

def main():
    args = parser.parse_args()
    random_trials = args.random
    lr = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    save_path = args.save
    load_model_path = args.load_model
    transfer_learning = args.transfer_learning

    print("Loading settings...")
    config = Config()
    config.load_from_file()
    run_dir = create_run_dir(config.output_dir)
    train_dir = config.train_data_dir
    val_dir = config.val_data_dir
    total_classes = config.data_classes

    # Load the dataset, and split the dataset to train, validation, and test
    print("Loading the dataset...")
    train_data, val_data = load_dataset(train_dir, val_dir)

    if load_model_path is not None:
        model = torch.load(load_model_path)
        model.eval()

        # TODO: Load test dataset and run the model on the test dataset
        return
    elif random_trials is not None:
        # Initialize  data loaders
        # Train the model
        for i in range(random_trials):
            lr, batch_size, epochs = hyperparam_randomizer()
            model = run_epoch(train_data, val_data, lr, batch_size, epochs, total_classes, run_dir[1], i=i)
    else:
        model = run_epoch(train_data, val_data, lr, batch_size, epochs, total_classes, run_dir[1])

    if save_path is not None:
        torch.save(model, save_path)


main()
