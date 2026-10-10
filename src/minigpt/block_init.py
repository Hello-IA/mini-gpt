import torch

from minigpt.init_matrice import init_Matrice
from minigpt.parametre import B, T, d, d_k, n_heads, K
def block_init():
    data = {"gamma1": torch.ones((d,), device = "cuda", requires_grad = True),
            "gamma2": torch.ones((d,), device = "cuda", requires_grad = True),

            "W_Q": init_Matrice(d, d),
            "W_K": init_Matrice(d, d),
            "W_V" : init_Matrice(d, d),
            "W_O" : init_Matrice(d, d),

            "W1": init_Matrice(d, 4*d),
            "W2": init_Matrice(4*d, d),
            
            "b1":torch.zeros((4*d,), device = "cuda", requires_grad = True),
            "b2":torch.zeros((d,), device = "cuda", requires_grad = True)}
    return data
