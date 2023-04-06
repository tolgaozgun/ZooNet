import logging
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageNet
from settings import Settings



# Training hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    logger = logging.getLogger(__name__)
    logger.info("Loading settings...")
    settings = Settings()
    settings.load_from_file()
    
    # Load the dataset, and split the dataset to train, validation, and test
    logger.info("Loading the dataset...")
    train_data = ImageNet('./data', split='train', transform=transforms.ToTensor())
    val_data = ImageNet('./data', split='val', transform=transforms.ToTensor())


    # Initialize  data loaders
    logger.info("Initializing data loaders...")
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE) 
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE) 

    # Calculate steps per epoch
    train_steps = len(trainDataLoader.dataset) // BATCH_SIZE
    val_steps = len(valDataLoader.dataset) // BATCH_SIZE

    # Train the model
    logger.info("Initializing the model...")
    # model = ZooNet(
    #     num_channels=3,
    #     classes=len(train_data.dataset.classes)
    # ).to(device)

if __name__ == '__main__':
    main()
