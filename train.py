from model import ZooNet
from torch.optim import Adam
from torch import nn
import time
import logging

logger = logging.getLogger(__name__)

# TODO: add optimizer and loss_fn options to parameters
def train(train_data_loader, validation_data_loader, train_steps: int, validation_steps: int, num_channels: int=3, classes: int=1, learning_rate:float=0.1, epochs: int=10):
    logger.info("Initializing the ZooNet Model...")
    model = ZooNet(num_channels, classes).to(device)

    logger.info("Initializing the optimizer and loss function")
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()
    
    # Update after each epoch
    training_history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    logger.info("Training the network...")
    start_time = time.time()

    for e in range(epochs):
        # Train mode
        model.train()

        train_loss = 0
        validation_loss = 0

        train_correct = 0
        validation_correct = 0

        # Loop over training set
        for (x, y) in training_data_loader:
            (x, y) = (x.to(device), y.to(device))

            # Forwards propagation
            prediction = model(x)
            loss = loss_fn(prediction, y)

            # Backwards propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update loss
            train_loss += loss
            train_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        # Loop over validation set
        with torch.no_grad():
            model.eval()

            for (x, y) in validation_data_loader:
                (x, y) (x.to(device), y.to(device))

                # Make prediction
                prediction = model(x)
                loss = loss_fn(predicton, y)
                validation_loss += loss

                # Calculate the number of correct predictions 
                validation_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        avg_train_loss = train_loss / train_steps
        avg_validation_loss = validation_loss / validatin_steps
        train_correct_accuracy = train_correct / len(train_data_loader.dataset)
        validation_correct_accuracy = train_correct / len(train_data_loader.dataset)

        training_history['train_loss'].append(avg_train_loss)
        training_history['train_accuracy'].append(train_correct_accuracy)
        training_history['validation_loss'].append(avg_validation_loss)
        training_history['validation_accuracy'].append(validation_correct_accuracy)

        logger.info(f"EPOCH: {e+1}/{epochs}")
        logger.info(f"\tTrain loss: {avg_train_loss}, Train accuracy: {train_correct_accuracy}")
        logger.info(f"\tValidation loss: {avg_validation_loss}, Validation accuracy: {validation_correct_accuracy}")

    end_time = time.time()
    logger.info("Finished training.")
    logger.info(f"Training duratin: {start_time - end_time}")
