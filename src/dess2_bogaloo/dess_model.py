from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DESSHead(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, raw_sigma = x.chunk(2, dim=-1)
        sigma = F.softplus(raw_sigma) + 1e-6
        return mu, sigma


class DessReranker(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, input_dim * 2),
        )
        self.head = DESSHead()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.backbone(x))
