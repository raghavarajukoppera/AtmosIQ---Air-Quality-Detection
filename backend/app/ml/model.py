from __future__ import annotations

import torch
from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        support = features @ self.weight
        return norm_adj @ support


class GCNRegressor(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        x = self.gcn1(features, norm_adj)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.gcn2(x, norm_adj)
        return x.squeeze(-1)


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))
