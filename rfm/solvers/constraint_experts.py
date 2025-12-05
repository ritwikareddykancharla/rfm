import torch
import torch.nn as nn


class ConstraintExpertMoE(nn.Module):
    """Placeholder constraint Mixture-of-Experts.

    In a full implementation, this would contain multiple experts for
    different constraint families (flow, capacity, timing, etc.).
    For now, we implement a simple MLP that returns a scalar penalty.
    """

    def __init__(self, hidden_dim: int, n_experts: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Compute a simple penalty from violations v.

        Args:
            v: (m,) violation vector.

        Returns:
            Scalar penalty tensor.
        """
        if v.ndim == 1:
            v = v.unsqueeze(0)
        # Summarize violations as mean and max
        stats = torch.stack([v.mean(dim=-1), v.max(dim=-1).values], dim=-1)
        return self.net(stats).sum()
