#!/usr/bin/python3

"Common utility functions for training/testing CNNs on the Fashion MNIST data"
import torch
import torchvision
from torchvision import transforms

def get_training_data_loader(batch_size: int = 256):
    "Common function for getting the data loader for model training"
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),      # Flip images horizontally
        transforms.RandomRotation(degrees=5),        # Slight rotations
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Slight shifts
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))         # Normalize to [-1, 1] range
    ])

    # Load the training and the test datasets
    traindata = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        traindata,
        batch_size=batch_size,
        shuffle=True,
        num_workers=24,
        pin_memory=True
    )

    return train_loader


def get_testing_data_loader(batch_size: int = 256):
    "Common function for getting the data loader for model testing"
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))         # Normalize to [-1, 1] range
    ])

    testdata = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=test_transform
    )

    test_loader = torch.utils.data.DataLoader(
        testdata, batch_size=batch_size, shuffle=False, num_workers=24
    )

    return test_loader


def evaluate_accuracy(model, loader, device):
    "Common function for evaluating the accuracy of image classification model"
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_x, test_y in loader:
            test_x, test_y = test_x.to(device), test_y.to(device)
            logits = model(test_x)
            y_pred = torch.argmax(logits, dim=1)
            correct += (y_pred == test_y).sum().item()
            total += test_y.size(0)
    return correct / total
