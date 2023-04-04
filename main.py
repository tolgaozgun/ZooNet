import logging
import torch
import torchvision
from model import ZooNet
from settings import Settings

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
    
    # Load the dataset, and split the dataset to train, validation, and test
    logger.info("Loading the dataset...")
    train_data = None
    test_data = None

    # Initialize  data loaders
    logger.info("Initializing data loaders...")
    train_data_loader = None # TODO:
    val_data_loader = None # TODO:

    # Calculate steps per epoch
    train_steps = len(trainDataLoader.dataset) // BATCH_SIZE
    val_steps = len(valDataLoader.dataset) // BATCH_SIZE

    # Train the model
    logger.info("Initializing the model...")
    model = ZooNet(
        num_channels=3,
        classes=len(train_data.dataset.classes)
    ).to(device)
