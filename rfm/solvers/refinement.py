import torch
import torch.nn as nn
import torch.autograd as autograd


class RefinementStep(nn.Module):
    """Simple gradient-style refinement step.

    This is a minimal differentiable refinement operator:
    x <- clip(x - eta * dL/dx, [0,1])
    """

    def __init__(self, eta: float = 0.1):
        super().__init__()
        self.eta = eta

    def forward(self, x: torch.Tensor, obj: torch.Tensor) -> torch.Tensor:
        grad, = autograd.grad(obj, x, retain_graph=True, allow_unused=True)
        if grad is None:
            return x
        x_new = x - self.eta * grad
        return torch.clamp(x_new, 0.0, 1.0)
