from minigpt.tokenizer import BPETokenizer
from minigpt.batch import get_batch
import torch
from minigpt.parametre import B, T, d, d_k, n_heads, K


def init_embedding(V, d):
    return torch.normal(0, 0.02, size = (V, d), requires_grad=True, device = "cuda")

def batch_embedding(X_b, E):
    return E[X_b, :]




