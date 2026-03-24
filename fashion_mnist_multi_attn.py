#!/usr/bin/python3
"Implementation of CNN model with Multi-Head Attention Mechanism for Fashion MNIST dataset"

import torch
from torch import nn

import common

class MultiHeadAttentionPool2d(nn.Module):
    "Implementation of multi-head attention mechanism for CNN"
    def __init__(self, channels: int, height: int, width: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        assert self.head_dim * heads == channels, "Channels must be divisible by heads"

        # 1. Learned Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, channels, height, width) * 0.02)

        # 2. Multi-Head Projections
        # Q reduces spatial size (stride=2), K and V stay at full resolution
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=2, stride=2)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.skip = nn.AvgPool2d(kernel_size=2)

        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        "Forward propagation for the Attention Mechanism"
        B, C, H, W = x.shape
        x_pos = x + self.pos_embed

        # Generate Q, K, V
        Q = self.q_conv(x_pos) # [B, C, H/2, W/2]
        K = self.k_conv(x_pos) # [B, C, H, W]
        V = self.v_conv(x)     # [B, C, H, W]

        B, _, H_out, W_out = Q.shape
        N_out = H_out * W_out
        N_in = H * W

        # Reshape for multi-head attention
        # Q: [B, heads, N_out, head_dim]
        # K: [B, heads, head_dim, N_in]
        # V: [B, heads, N_out, head_dim]
        Q = Q.view(B, self.heads, self.head_dim, N_out).permute(0, 1, 3, 2)
        K = K.view(B, self.heads, self.head_dim, N_in)
        V = V.view(B, self.heads, self.head_dim, N_in).permute(0, 1, 3, 2)

        # Scaled Dot-Product Attention
        attn = torch.matmul(Q, K) * self.scale # [B, heads, N_out, N_in]
        attn = torch.softmax(attn, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, V) # [B, heads, N_out, head_dim]

        # Re-assemble heads
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H_out, W_out)
        out = self.out_conv(out)

        # Residual + Norm
        return self.norm(out + self.skip(x))


class CNNImageClassifier(nn.Module):
    "Implementation of CNN model with Multi-Head Attention Mechanism"
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            MultiHeadAttentionPool2d(32, 28, 28, heads=4),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            MultiHeadAttentionPool2d(64, 14, 14, heads=8)
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
        "Forward propagation for this CNN model"
        return self.network(self.features(x))


def train_model(model, train_loader, device, num_epochs: int = 10):
    "Implementation of the model training for CNN with Multi-Head Attention Mechanism"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # CosineAnnealingLR: Gradually reduces LR to a minimum (eta_min) over T_max epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

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

    model = CNNImageClassifier().to(device)
    train_loader = common.get_training_data_loader()
    train_model(model, train_loader, device, num_epochs=40)

    # Evaluate on test set
    test_loader = common.get_testing_data_loader()
    test_acc = common.evaluate_accuracy(model, test_loader, device)
    print(f"Accuracy on test data: {test_acc*100:.2f}%")


if __name__ == '__main__':
    # Call the main function
    main()
