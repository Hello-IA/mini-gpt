import torch


def init_Matrice(A, B):
    return torch.normal(0, 0.02, size = (A, B), requires_grad=True, device = "cuda")

