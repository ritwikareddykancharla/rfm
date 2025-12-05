import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftRelaxation(nn.Module):
    """Temperature-controlled sigmoid relaxation for binary variables."""

    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits / self.tau)
