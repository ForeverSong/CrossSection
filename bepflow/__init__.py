"""bepflow: GPU (PyTorch) prototypes for two-phase backward erosion piping.

darcy_pipe  -- Robbins-type steady Darcy + pipe model, batched Monte Carlo
pipe_el2d   -- 2-D Euler-Lagrange pipe-segment solver (NTLP algorithm in torch)
"""
import torch


def pick_device(prefer="cuda"):
    """'cuda' when a GPU is visible to torch, otherwise 'cpu'."""
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
