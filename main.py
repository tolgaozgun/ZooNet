import logging
import copy
import os
import argparse
from random import randint
from PIL import Image

import torch
from torch.utils.data import random_split, DataLoader
from torchvision.utils import save_image
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
from config import Config
from matplotlib import pyplot as plt

from train import train
from utils import create_run_dir, hyperparam_randomizer, training_history_to_excel, class_name_to_real_name
from dataset import load_dataset, unnormalize_im, CustomDataset, idx_to_class


# Hyperparameter constants
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
mean=[0.4914, 0.4822, 0.4465]
std=[0.2023, 0.1994, 0.2010]

# Command line args
parser = argparse.ArgumentParser(
    prog="ZooNet",
    description="Command line interface for ZooNet.",
    epilog=""
)
parser.add_argument("-r", "--random", help="Randomly creates hyperparameters for the current run." + 
                    " If this option is selected any other flags providing a value to hyperparameters will be ignored",
                    type=int, default=None)
parser.add_argument("-l", "--learning-rate", help="Learning rate for the current run.", type=float, default=LEARNING_RATE)
parser.add_argument("-b", "--batch-size", help="Batch size for the current run.", type=int, default=BATCH_SIZE)
parser.add_argument("-e", "--epochs", help="Number of epochs for this run.", type=int, default=EPOCHS)
parser.add_argument("-s", "--save", help="Saves the learned model parameters to the given path.", type=str, default=None)
parser.add_argument("--load-model", help="Loads the model parameters for running tests. No learning is made if this flag is true, the model is run on the test dataset.", type=str, default=None)
parser.add_argument("-t", "--transfer-learning", help="Enables transfer learning, runs the pretrained model on thedataset.")
parser.add_argument("-o", "--output", type=str, default=None)


def run_epoch(train_data_loader, val_data_loader, lr, batch_size, epochs, total_classes, run_dir, i=None):
    print(f"Starting current run...\n\tLearning rate: {lr}, Batch size: {batch_size}, Total epochs: {epochs}")
    
    # Calculate steps per epoch
    train_steps = len(train_data_loader.dataset) // batch_size
    val_steps = len(val_data_loader.dataset) // batch_size
    print(f"\tSize of the train set: {len(train_data_loader.dataset)}")
    print(f"\tSize of the validation set: {len(val_data_loader.dataset)}")

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

    return model, training_history, duration


def run_test(model, test_data_loader, run_dir=None):
    training_history = {}
    model.eval()  # Set the model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    test_loss = 0.0
    test_accuracy = 0.0
    total_samples = len(test_data_loader.dataset)
    results = []
    acc_ims = []
    inacc_ims = []

    with torch.no_grad():
        for batch_data, batch_labels in test_data_loader:
            batch_size = len(batch_data)
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            pred = model(batch_data)
            loss = criterion(pred, batch_labels)

            test_loss += loss
            _, predicted = torch.max(pred, 1)
            acc_count = (predicted == batch_labels).sum().item()
            test_accuracy += acc_count

            if acc_count > 0:
                # Get the index of the first accurately predicted example in the batch
                index = (predicted == batch_labels).nonzero(as_tuple=False)[0].item()
                inacc_index = (predicted != batch_labels).nonzero(as_tuple=False)[0].item()

                # Get the predicted and ground truth labels
                predicted_label = predicted[index].item()
                true_label = batch_labels[index].item()

                inacc_predicted_label = predicted[inacc_index].item()
                inacc_true_label = batch_labels[inacc_index].item()

                # Get the corresponding image and move it to the CPU
                im = batch_data[index].cpu().detach()
                inacc_true_im = batch_data[inacc_index].cpu().detach() 

                # Display the image and labels
                # image = np.transpose(image, (1, 2, 0))
                acc_ims.append({
                    "image": im,
                    "predicted": idx_to_class[predicted_label],
                    "true": idx_to_class[true_label]
                })
                inacc_ims.append({
                    "image": inacc_true_im,
                    "predicted": idx_to_class[inacc_predicted_label],
                    "true": idx_to_class[inacc_true_label]
                })


    test_loss /= total_samples
    test_accuracy /= total_samples
    test_accuracy_percentage = test_accuracy * 100

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy_percentage:.2f}%")
    return test_loss, test_accuracy, acc_ims, inacc_ims


def main():
    args = parser.parse_args()
    random_trials = args.random
    lr = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    save_path = args.save
    load_model_path = args.load_model
    transfer_learning = args.transfer_learning
    output_path = args.output

    print("Loading settings...")
    config = Config()
    config.load_from_file()
    run_number, run_dir = create_run_dir(config.output_dir, test=True if load_model_path is not None else False)
    train_dir = config.train_data_dir
    # test_dir = config.test_data_dir
    val_dir = config.val_data_dir
    total_classes = config.data_classes

    if output_path is not None:
        run_dir = output_path

    class_to_real_map = class_name_to_real_name("folder_to_class.txt")

    # Load the dataset, create data loaders
    print("Loading the dataset...")
    normalize_transform = transforms.Normalize(
        mean=mean,
        std=std,
    )
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize_transform
    ])
    train_data = CustomDataset(train_dir, transform)
    train_len = int(len(train_data) * 0.9)
    test_len = len(train_data) - train_len
    train_data, test_data = torch.utils.data.random_split(train_data, [train_len, test_len])
    val_data = CustomDataset(val_dir, transform)
    # train_data, val_data, test_data = load_dataset(train_dir, val_dir)

    # Load an existing model and test it on the test data
    if load_model_path is not None:
        print(f"Loading the model at {load_model_path}")
        model = torch.load(load_model_path)
        model.eval()

        # Freeze layers
        for param in model.parameters():
            param.requires_grad = False

        print("Initializing data loaders...")
        test_loader = DataLoader(test_data , batch_size=batch_size, shuffle=False, 
                               num_workers=4, drop_last=True)
        print(f"Size of the test set: {len(test_loader.dataset)}")
        
        # Run the test
        test_loss, test_accuracy, acc_ims, inacc_ims = run_test(model, test_loader, run_dir)

        # Display some of the images
        for img_dict in acc_ims:
            im_arr = unnormalize_im(img_dict["image"])
            true_label = class_to_real_map[img_dict["true"]]
            predicted_label = class_to_real_map[img_dict["predicted"]]

            # im = Image.fromarray(im_arr).convert('RGB')
            save_image(im_arr, f"{run_dir}/acc_ims/true_{true_label}_predicted_{predicted_label}.png")
        for img_dict in inacc_ims:
            true_im_arr = unnormalize_im(img_dict["image"])
            true_label = class_to_real_map[img_dict["true"]]
            predicted_label = class_to_real_map[img_dict["predicted"]]

            # im = Image.fromarray(im_arr).convert('RGB')
            im = true_im_arr.numpy().transpose(1, 2, 0)
            fig, ax = plt.subplots()
            ax.imshow(im)
            ax.set_title(f"True label: {true_label}, Predicted label: {predicted_label}")
            fig.savefig(f"{run_dir}/inacc_ims/true_{true_label}_predicted_{predicted_label}.png")
            plt.close(fig)
            
            # save_image(true_im_arr, f"{run_dir}/inacc_ims/true_{true_label}_predicted_{predicted_label}.png")


    # Run with randomly generated params
    elif random_trials is not None:
        print("Initializing data loaders...")
        train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) 
        val_data_loader = DataLoader(val_data, batch_size=batch_size) 

        # Train the model
        for i in range(random_trials):
            lr, batch_size, epochs = hyperparam_randomizer()
            model, training_history, duration = run_epoch(train_data_loader, val_data_loader, lr, batch_size, epochs, total_classes, run_dir, i=i)
    # Run with command line params
    else:
        print("Initializing data loaders...")
        train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size) 
        val_data_loader = DataLoader(val_data, batch_size=batch_size) 
        model, training_history, duration = run_epoch(train_data_loader, val_data_loader, lr, batch_size, epochs, total_classes, run_dir)

    # Plot results
    if load_model_path:
        pass
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plot_range = range(1, epochs+1)
        
        # Accuracy
        ax1.set_title("Train vs. validation dataset accuracy")
        ax1.xaxis.set_ticks(plot_range)

        ax1.plot(plot_range, training_history["train_accuracy"], color="red", label="Train accuracy")
        ax1.plot(plot_range, training_history["validation_accuracy"], color="blue", label="Validation accuracy")

        # Loss
        ax2.set_title("Train vs. validation loss")
        ax2.xaxis.set_ticks(plot_range)

        ax2.plot(plot_range, training_history["train_loss"], color="red", label="Train loss")
        ax2.plot(plot_range, training_history["validation_loss"], color="blue", label="Validation loss")
        
        fig.legend()
        fig.savefig(f"{run_dir}/plots.png")
    if save_path is not None and save_path != "" and save_path:
        torch.save(model, save_path)

if __name__ == "__main__":
    main()
