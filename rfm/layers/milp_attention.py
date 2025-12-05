import torch
import torch.nn as nn
import torch.nn.functional as F


class MILPAttention(nn.Module):
    """MILP-aware self-attention.

    Given variable embeddings X and dual-like scores h = A^T v, this layer
    computes attention logits that are penalized for variables involved in
    heavily violated constraints.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, gamma: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.gamma = gamma

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Apply MILP-aware attention.

        Args:
            x: (n, d) variable embeddings.
            h: (n,) dual-like scores derived from A^T v.

        Returns:
            Updated embeddings of shape (n, d).
        """
        n, d = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (n, H, d_head)
        H = self.n_heads
        d_head = d // H
        q = q.view(n, H, d_head)
        k = k.view(n, H, d_head)
        v = v.view(n, H, d_head)

        # Scaled dot-product attention
        logits = torch.einsum("nhd,mhd->hnm", q, k) / (d_head ** 0.5)  # (H, n, n)

        # Inject dual-informed penalty: broadcast h over rows
        if h is not None:
            h_penalty = self.gamma * h  # (n,)
            logits = logits - h_penalty.view(1, 1, n)

        attn = F.softmax(logits, dim=-1)
        out = torch.einsum("hnm,mhd->nhd", attn, v).reshape(n, d)

        return self.out_proj(out) + x  # residual
