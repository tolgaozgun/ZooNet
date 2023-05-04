from model_v3 import ZooNet
from torch import nn
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: add optimizer and loss_fn options to parameters
def train(train_data_loader, val_data_loader, train_steps: int, validation_steps: int,
            classes: int=1, learning_rate:float=0.01, epochs: int=10):
    print("Initializing the ZooNet Model...")

    cuda_or_cpu = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Currently using: {cuda_or_cpu}. Version: {torch.version.cuda}")
    model = ZooNet(classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)
    loss_fn = nn.CrossEntropyLoss()  
    
    # Update after each epoch
    training_history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": []
    }

    print("Training the network...")
    start_time = time.time()

    for e in range(epochs):
        # Train mode
        model.train()

        train_loss = 0
        validation_loss = 0

        train_correct = 0
        validation_correct = 0

        # Loop over training set
        for (x, y) in train_data_loader:
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

            for (x, y) in val_data_loader:
                (x, y) = (x.to(device), y.to(device))

                # Make prediction
                prediction = model(x)
                loss = loss_fn(prediction, y)
                validation_loss += loss

                # Calculate the number of correct predictions 
                validation_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

        avg_train_loss = train_loss / train_steps
        avg_validation_loss = validation_loss / validation_steps
        train_correct_accuracy = train_correct / len(train_data_loader.dataset)
        validation_correct_accuracy = validation_correct / len(val_data_loader.dataset)

        training_history['train_loss'].append(avg_train_loss)
        training_history['train_accuracy'].append(train_correct_accuracy)
        training_history['validation_loss'].append(avg_validation_loss)
        training_history['validation_accuracy'].append(validation_correct_accuracy)

        print(f"EPOCH: {e+1}/{epochs}")
        print(f"\tTrain loss: {avg_train_loss}, Train accuracy: {train_correct_accuracy}")
        print(f"\tValidation loss: {avg_validation_loss}, Validation accuracy: {validation_correct_accuracy}")


    end_time = time.time()
    print("Finished training.")
    duration = end_time - start_time
    print(f"Training duration: {duration}")

    torch.save(model, './out')
    return training_history, duration


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res