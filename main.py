import logging
import torch
import torchvision
from settings import Settings

def main():



if __name__ == '__main__':
    main()


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
    logger.info("Loading settings...")
    settings = Settings()
    settings.load_from_file()
    
    # Load the dataset, and split the dataset
    logger.info("Loading the dataset...")

    
    

