import argparse
import torch
import logging

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True, 
        help="Path to output trained model.")
ap.add_argument("-p", "--plot", type=str, required=True, 
        help="Path to output loss/accuracy plot")

# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

# Data split
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    logger = logging.getLogger(__name__)
    args = vars(ap.parse_args())
    
    # Load the dataset, and split the dataset
    logger.info("Loading the dataset...")

    
    

main()