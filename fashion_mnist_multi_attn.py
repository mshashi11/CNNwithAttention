#!/usr/bin/python3
"Implementation of CNN model with Multi-Head Attention Mechanism for Fashion MNIST dataset"

import time
import torch
from torch import nn

import common

class MultiHeadAttentionPool2d(nn.Module):
    "Implementation of multi-head attention mechanism for CNN"
    def __init__(self, channels: int, height: int, width: int, heads: int = 8):
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
        self.norm = nn.BatchNorm2d(channels)
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
        Q = Q.view(B, self.heads, self.head_dim, N_out).permute(0, 1, 3, 2)
        K = K.view(B, self.heads, self.head_dim, N_in)
        V = V.view(B, self.heads, self.head_dim, N_in).permute(0, 1, 3, 2)

        # ScalDot-Product Attention
        attn = torch.matmul(Q, K) * self.scale # [B, heads, N_out, N_in]
        attn = torch.softmax(attn, dim=-1)

        # Weighted sum of values
        out = torch.matmul(attn, V) # [B, heads, N_out, head_dim]

        # Re-assemble heads
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H_out, W_out)
        out = self.out_conv(out)

        # Residual + Norm
        return self.norm(out + self.skip(x))


class GlobalTransformerBlock(nn.Module):
    """
    Final reasoning layer: Every 7x7 patch talks to every other patch.
    """
    def __init__(self, channels: int, heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, 49, channels) * 0.02)
        # PyTorch's optimized MultiheadAttention
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape image to sequence: [B, 49, C]
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        x_flat = x_flat + self.pos_embed

        # Attention + Residual (Pre-Norm)
        attn_out, _ = self.mha(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
        x_flat = x_flat + attn_out

        # Feed Forward + Residual (Pre-Norm)
        ffn_out = self.ffn(self.norm2(x_flat))
        x_flat = x_flat + ffn_out

        # Back to image shape: [B, C, 7, 7]
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


class CNNImageClassifier(nn.Module):
    "Implementation of CNN model with Multi-Head Attention Mechanism"
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 72, 3, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout2d(0.02),
            nn.Conv2d(72, 72, 3, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout2d(0.02),
            nn.Conv2d(72, 72, 3, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout2d(0.02),
            nn.Conv2d(72, 72, 3, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout2d(0.02),
            nn.Conv2d(72, 72, 3, padding=1), # Added 5th layer in Block 1 for Exp 101
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout2d(0.02),
            MultiHeadAttentionPool2d(72, 28, 28, heads=6),

            nn.Conv2d(72, 184, 3, padding=1),
            nn.BatchNorm2d(184),
            nn.ReLU(),
            nn.Dropout2d(0.02),
            nn.Conv2d(184, 184, 3, padding=1), # Added 2nd layer in Block 2 for Exp 85
            nn.BatchNorm2d(184),
            nn.ReLU(),
            nn.Dropout2d(0.02),
            MultiHeadAttentionPool2d(184, 14, 14, heads=4),

            # Reasoning Global Block
            GlobalTransformerBlock(184, heads=4)

        )
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*184, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.20),
            nn.GELU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        "Forward propagation for this CNN model"
        return self.network(self.features(x))


def train_model(model, train_loader, device, num_epochs: int = 60, time_budget_sec=600):
    "Implementation of the model training for CNN with Multi-Head Attention Mechanism"
    start_time = time.time()
    initial_lr = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

    # CosineAnnealingLR with T_max adjusted to expected epoch count
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)
    
    # LR Warmup
    warmup_epochs = 5

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(num_epochs):
        # Adjust LR for warmup
        if epoch < warmup_epochs:
            lr = initial_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        model.train()
        epoch_loss = 0
        batch_idx = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Add Gaussian Noise during training
            batch_x = batch_x + torch.randn_like(batch_x) * 0.02
            
            optimizer.zero_grad()
            y = model(batch_x)
            loss = criterion(y, batch_y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            cur_time = time.time()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Time so far: {cur_time - start_time:.2f}s")
            batch_idx += 1
            # Check if the time budget has expired
            if cur_time - start_time >= time_budget_sec:
                print("Exiting training since training time budget exceeded")
                return

        scheduler.step()
        print(f"Epoch: {epoch} | Loss: {epoch_loss:.2f}")


def main():
    "Main function of the script, execution starts here"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = CNNImageClassifier().to(device)
    train_loader = common.get_training_data_loader(batch_size=256)
    start_time = time.time()
    train_model(model, train_loader, device, num_epochs=60, time_budget_sec=600)
    end_time = time.time()

    # Evaluate on test set
    test_loader = common.get_testing_data_loader()
    test_acc = common.evaluate_accuracy(model, test_loader, device)
    print("---Summary---")
    print(f"Accuracy: {test_acc*100:.2f}\nTime: {(end_time - start_time):.2f}")


if __name__ == '__main__':
    # Call the main function
    main()
