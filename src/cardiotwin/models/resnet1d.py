from __future__ import annotations
import torch
import torch.nn as nn

class BasicBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))

class TinyResNet1D(nn.Module):
    """CPU-friendly deep model scaffold for 12-lead ECG.

    Input: (batch, samples, leads) or (batch, leads, samples)
    Output: logits for multi-label classification.
    """
    def __init__(self, n_leads: int = 12, n_classes: int = 5, channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, channels, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            BasicBlock1D(channels),
            nn.MaxPool1d(2),
            BasicBlock1D(channels),
            nn.MaxPool1d(2),
            BasicBlock1D(channels),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, n_classes),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected 3D tensor")
        if x.shape[1] != 12 and x.shape[2] == 12:
            x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
