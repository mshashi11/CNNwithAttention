#!/usr/bin/python3
"Implementation of CNN with Single-Head Attention Mechanism for the Fashion MNIST dataset"

import torch
from torch import nn

import common

class AttentionPool2d(nn.Module):
    """
    Self-Attention pooling with Positional Encodings and Residuals.
    """
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        # 1. Learned Positional Encoding: Allows the network to know 'where' pixels are.
        # Shape matches the input feature map: [1, Channels, H, W]
        self.pos_embed = nn.Parameter(torch.randn(1, channels, height, width) * 0.02)

        self.query_conv = nn.Conv2d(channels, channels, kernel_size=2, stride=2)
        self.key_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # 2. Normalization: Essential for attention stability
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)

        # 3. Residual Path: Preserves the original signal using standard downsampling
        self.skip = nn.AvgPool2d(kernel_size=2)

        self.scale = channels ** -0.5

    def forward(self, x):
        "Forward propagation on the input tensor for the Attention mechanism"
        B, C, H, W = x.shape

        # Inject positional information before projecting Q, K, V
        x_pos = x + self.pos_embed

        Q = self.query_conv(x_pos)
        B, C, H_out, W_out = Q.shape

        Q_flat = Q.view(B, C, -1).transpose(1, 2)
        K_flat = self.key_conv(x_pos).view(B, C, -1)
        V_flat = self.value_conv(x).view(B, C, -1).transpose(1, 2)

        # Standard Attention computation
        attn = torch.bmm(Q_flat, K_flat) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(attn, V_flat)
        out = out.transpose(1, 2).view(B, C, H_out, W_out)
        out = self.out_conv(out)

        # Apply Residual + Norm (x_skip + Attention_out)
        return self.norm(out + self.skip(x))


class CNNImageClassifier(nn.Module):
    "Implementation of CNN model with Single-Head Attention Mechanism"
    def __init__(self, num_classes=10):
        super().__init__()

        # Extract features using Convolutional and Attention Layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            AttentionPool2d(channels=32, height=28, width=28),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            AttentionPool2d(channels=64, height=14, width=14)
        )

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*64, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.10),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.10),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        "Forward propagation for the CNN model"
        return self.network(self.features(x))


def train_model(model, train_loader, device, num_epochs: int = 30):
    "Training the model for CNN with Single-Head Attention Mechanism"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            y = model(batch_x)
            loss = criterion(y, batch_y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch} | Loss: {epoch_loss:.2f}")


def main():
    "Main function of the script, execution starts here"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = common.get_training_data_loader()
    model = CNNImageClassifier().to(device)
    train_model(model, train_loader, device, num_epochs=30)

    test_loader = common.get_testing_data_loader()
    test_acc = common.evaluate_accuracy(model, test_loader, device)
    print(f"Accuracy on test data: {test_acc*100:.2f}%")


if __name__ == '__main__':
    # Call the main function
    main()
