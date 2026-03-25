#!/usr/bin/python3
"An implementation of basic CNN with PyTorch for the Fashion MNIST dataset"

import torch
from torch import nn

import common

class CNNImageClassifier(nn.Module):
    "CNN based model for image classification"
    def __init__(self, num_classes = 10):
        super().__init__()
        # This generates the features for classification using convolution/pooling operations
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Image now becomes 14*14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Resulting image is 7*7
            nn.MaxPool2d(kernel_size=2)
        )

        # A deep learning network for image classification
        self.network = nn.Sequential(
            # Flatten the image into a linear tensor
            nn.Flatten(),
            nn.Linear(in_features=7*7*64, out_features=256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.10),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.Dropout(0.10),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        "Run forward propagation for the input tensor"
        return self.network(self.features(x))


def train_model(model, train_loader, device, num_epochs: int = 10):
    "Train the given model on the training data"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            # Do forward pass on the given batch
            y = model(batch_x)

            # Compute the loss function
            loss = criterion(y, batch_y)
            epoch_loss += loss

            # Backward propagation
            loss.backward()

            # Gradient descent
            optimizer.step()

        print(f"Epoch: {epoch} | Loss: {epoch_loss:.2f}")


def main():
    "Main function of the program, execution starts here"
    # Run the model training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = common.get_training_data_loader()
    test_loader = common.get_testing_data_loader()

    cnn_model = CNNImageClassifier().to(device)
    train_model(cnn_model, train_loader, device, num_epochs=40)

    test_accuracy = common.evaluate_accuracy(cnn_model, test_loader, device)
    print(f"Model accuracy on test data = {test_accuracy*100:.2f}%")


if __name__ == "__main__":
    # Call the main function
    main()
