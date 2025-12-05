"""Toy script to run MILP-Transformer on a random synthetic MILP.

This is not a serious benchmark, just a smoke test to verify that
the repository and model are wired correctly.
"""
import torch
from rfm.training.trainer import RFMTrainer


def main():
    n_vars = 32
    m_cons = 16
    trainer = RFMTrainer(n_vars=n_vars, device="cpu")

    # Random toy MILP: A x <= b, x in [0,1]
    A = torch.randn(m_cons, n_vars)
    b = torch.randn(m_cons)
    c = torch.randn(n_vars)

    for step in range(5):
        obj, aux = trainer.step(A, b, c)
        print(f"Step {step}: obj = {obj:.4f}")


if __name__ == "__main__":
    main()
