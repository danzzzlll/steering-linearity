import torch
import torch.nn as nn


class SteerMLP(nn.Module):
    def __init__(self, num_features, dropout=0.1):
        super().__init__()
        mid = num_features // 2

        self.layer_1 = nn.Linear(num_features, mid)
        self.norm1   = nn.LayerNorm(mid)
        self.drop1   = nn.Dropout(dropout)

        self.layer_2 = nn.Linear(mid, num_features)
        self.norm2   = nn.LayerNorm(num_features)
        self.drop2   = nn.Dropout(dropout)

        self.layer_3 = nn.Linear(num_features, num_features)
        self.gate    = nn.Linear(num_features, num_features)

    def forward(self, x):
        identity = x

        x = self.layer_1(x)
        x = self.norm1(x)
        x = torch.nn.functional.gelu(x)
        x = self.drop1(x)

        x = self.layer_2(x)
        x = self.norm2(x)
        x = torch.nn.functional.gelu(x)
        x = self.drop2(x)

        delta = self.layer_3(x)
        gate  = torch.sigmoid(self.gate(x))
        return identity + gate * delta