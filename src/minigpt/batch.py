import torch
import torch.nn as nn

def get_batch(text, B, T, device):
    N = len(text)
    b_i = torch.randint(0, N -T -1, (B,), device = device)
    offsets = torch.arange(T, device = device)
    idex = b_i[:, None] + offsets[None, :]
    x = text[idex]
    y = text[idex+1]
    return x, y
x, y = get_batch(torch.randint(0, 10000, (500,), device = "cuda"), 5, 20, "cuda")
