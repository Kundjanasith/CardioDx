from __future__ import annotations
import math
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dropout=0.1):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel=7, dropout=0.1):
        super().__init__()
        self.f = nn.Sequential(ConvBlock(ch, ch, kernel=kernel, dropout=dropout), ConvBlock(ch, ch, kernel=kernel, dropout=dropout))
    def forward(self, x):
        return x + self.f(x)

class ResNet1DLite(nn.Module):
    def __init__(self, in_leads=12, n_classes=5, base=32, dropout=0.15):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_leads, base, kernel=15, stride=2, dropout=dropout),
            ResidualBlock(base, kernel=15, dropout=dropout),
            ConvBlock(base, base*2, kernel=9, stride=2, dropout=dropout),
            ResidualBlock(base*2, kernel=9, dropout=dropout),
            ConvBlock(base*2, base*4, kernel=7, stride=2, dropout=dropout),
            ResidualBlock(base*4, kernel=7, dropout=dropout),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(base*4, n_classes))
    def forward(self, x):
        return self.head(self.encoder(x))

class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(9, 19, 39), dropout=0.1):
        super().__init__()
        branch = out_ch // len(kernels)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, branch, k, padding=k//2, bias=False),
                nn.BatchNorm1d(branch),
                nn.SiLU(),
            ) for k in kernels
        ])
        self.proj = nn.Sequential(nn.Conv1d(branch*len(kernels), out_ch, 1), nn.BatchNorm1d(out_ch), nn.SiLU(), nn.Dropout(dropout))
    def forward(self, x):
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))

class InceptionTimeLite(nn.Module):
    def __init__(self, in_leads=12, n_classes=5, base=48, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            InceptionBlock1D(in_leads, base, dropout=dropout),
            nn.MaxPool1d(2),
            InceptionBlock1D(base, base*2, dropout=dropout),
            nn.MaxPool1d(2),
            InceptionBlock1D(base*2, base*2, dropout=dropout),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(base*2, n_classes))
    def forward(self, x):
        return self.head(self.net(x))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ECGTransformerLite(nn.Module):
    def __init__(self, in_leads=12, n_classes=5, d_model=96, nhead=4, depth=2, dropout=0.15):
        super().__init__()
        self.patch = nn.Conv1d(in_leads, d_model, kernel_size=25, stride=10, padding=12)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True, activation='gelu')
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))
    def forward(self, x):
        z = self.patch(x).transpose(1, 2)
        z = self.pos(z)
        z = self.encoder(z)
        return self.head(z.mean(dim=1))

def make_deep_model(name: str, in_leads=12, n_classes=5):
    name = name.lower()
    if name in {"resnet", "resnet1d", "resnet1d_lite"}:
        return ResNet1DLite(in_leads=in_leads, n_classes=n_classes)
    if name in {"inception", "inceptiontime", "inceptiontime_lite"}:
        return InceptionTimeLite(in_leads=in_leads, n_classes=n_classes)
    if name in {"transformer", "ecgtransformer", "transformer_lite"}:
        return ECGTransformerLite(in_leads=in_leads, n_classes=n_classes)
    raise ValueError(f"Unknown deep model: {name}")
