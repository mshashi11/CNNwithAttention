#!/usr/bin/python3
"Implementation of a generic CNN model with Multi-Head Attention Mechanisms"

import time
import torch
from torch import nn
import torchvision
from torchvision import transforms

import common

class MultiHeadAttentionPool2d(nn.Module):
    "Implementation of multi-head attention mechanism for CNN spatial downsampling"
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


class GlobalTransformerBlock(nn.Module):
    """
    Final reasoning layer: Every patch talks to every other patch.
    Generic version supporting arbitrary spatial dimensions.
    """
    def __init__(self, channels: int, height: int, width: int, heads: int = 16, dropout: float = 0.1):
        super().__init__()
        self.seq_len = height * width
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, channels) * 0.02)
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
        # Reshape image to sequence: [B, H*W, C]
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        x_flat = x_flat + self.pos_embed

        # Attention + Residual (Pre-Norm)
        attn_out, _ = self.mha(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))
        x_flat = x_flat + attn_out

        # Feed Forward + Residual (Pre-Norm)
        ffn_out = self.ffn(self.norm2(x_flat))
        x_flat = x_flat + ffn_out

        # Back to image shape: [B, C, H, W]
        return x_flat.permute(0, 2, 1).view(B, C, H, W)


class CNNImageClassifier(nn.Module):
    "Generic CNN model with Multi-Head Attention Mechanism"
    def __init__(self, in_channels=1, img_size=28, num_classes=10):
        super().__init__()
        
        # Track spatial size
        curr_s = img_size
        
        # Block 1: Deep Feature Extraction (5 layers)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 72, 3, padding=1),
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
            nn.Conv2d(72, 72, 3, padding=1),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Dropout2d(0.02),
        )
        
        # Pool 1
        self.pool1 = MultiHeadAttentionPool2d(72, curr_s, curr_s, heads=6)
        curr_s //= 2
        
        # Block 2: Higher Level Features (2 layers)
        self.block2 = nn.Sequential(
            nn.Conv2d(72, 184, 3, padding=1),
            nn.BatchNorm2d(184),
            nn.ReLU(),
            nn.Dropout2d(0.02),
            nn.Conv2d(184, 184, 3, padding=1),
            nn.BatchNorm2d(184),
            nn.ReLU(),
            nn.Dropout2d(0.02),
        )
        
        # Pool 2
        self.pool2 = MultiHeadAttentionPool2d(184, curr_s, curr_s, heads=4)
        curr_s //= 2
        
        # Global Reasoning Block
        self.global_block = GlobalTransformerBlock(184, curr_s, curr_s, heads=4)
        
        # Classification Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(curr_s * curr_s * 184, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.20),
            nn.GELU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        "Forward propagation"
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.global_block(x)
        return self.head(x)


def get_data_loaders(dataset_name='fashion_mnist', batch_size=256):
    "Generic function to get data loaders for different datasets"
    if dataset_name == 'fashion_mnist':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)
        
        in_channels = 1
        img_size = 28
        num_classes = 10
        
    elif dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        
        in_channels = 3
        img_size = 32
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, in_channels, img_size, num_classes


def train_model(model, train_loader, device, num_epochs=60, time_budget_sec=600):
    "Generic training loop"
    start_time = time.time()
    initial_lr = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-6)
    
    warmup_epochs = 5
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr = initial_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        model.train()
        epoch_loss = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
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
            
            if cur_time - start_time >= time_budget_sec:
                print("Exiting training since training time budget exceeded")
                return

        scheduler.step()
        print(f"Epoch: {epoch} | Loss: {epoch_loss:.2f}")


def main():
    "Main function"
    import argparse
    parser = argparse.ArgumentParser(description='Train generic CNN-Attention model')
    parser.add_argument('--dataset', type=str, default='fashion_mnist', choices=['fashion_mnist', 'cifar10'], help='Dataset to use')
    parser.add_argument('--batch_size', type=str, default='256', help='Batch size')
    args = parser.parse_args()
    
    batch_size = int(args.batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {args.dataset}")

    train_loader, test_loader, in_channels, img_size, num_classes = get_data_loaders(args.dataset, batch_size)
    
    model = CNNImageClassifier(in_channels=in_channels, img_size=img_size, num_classes=num_classes).to(device)
    
    start_time = time.time()
    train_model(model, train_loader, device, num_epochs=60, time_budget_sec=600)
    end_time = time.time()

    test_acc = common.evaluate_accuracy(model, test_loader, device)
    print("---Summary---")
    print(f"Accuracy: {test_acc*100:.2f}\nTime: {(end_time - start_time):.2f}")


if __name__ == '__main__':
    main()
