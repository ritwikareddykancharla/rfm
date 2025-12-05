import torch


def feasibility_violation(A: torch.Tensor, b: torch.Tensor, x: torch.Tensor) -> float:
    v = torch.relu(A @ x - b)
    return float(v.sum().item())
