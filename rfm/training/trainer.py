import torch
from torch.optim import Adam
from ..models.milp_transformer import MILPTransformer


class RFMTrainer:
    """Very lightweight training loop skeleton for RFM experiments."""

    def __init__(self, n_vars: int, lr: float = 1e-3, device: str = "cpu"):
        self.device = device
        self.model = MILPTransformer(n_vars=n_vars).to(device)
        self.opt = Adam(self.model.parameters(), lr=lr)

    def step(self, A, b, c):
        A = A.to(self.device)
        b = b.to(self.device)
        c = c.to(self.device)

        self.opt.zero_grad()
        x, aux = self.model(A, b, c)
        obj = (c * x).sum()
        obj.backward()
        self.opt.step()
        return obj.item(), aux
