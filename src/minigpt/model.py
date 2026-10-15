import torch
import torch.nn as nn
from minigpt.parametre import V, d, T, nb_block, device
from minigpt.block_init import block_init
from minigpt.embedding import init_embedding, batch_embedding
from minigpt.init_matrice import init_Matrice
from minigpt.rms_norm import RMSNorm
from minigpt.attention import attention
from minigpt.mlp import mlp
import numpy as np
from minigpt.batch import get_batch
class LLM_hugo(nn.Module):
    def __init__(self):
        super().__init__()
        self.E = nn.Parameter(init_embedding(V, d))
        self.P = nn.Parameter(init_embedding(T, d))

        self.W_out = nn.Parameter(init_Matrice(d, V))
        self.gamma_final = nn.Parameter(torch.ones((d,), device = device, requires_grad = True))


        self.blocks = nn.ModuleList()
        for n in range(nb_block):
            params = block_init()
            self.blocks.append(nn.ParameterDict({k: nn.Parameter(v) for k, v in params.items()}))
    def forward(self, x):
        t = x.shape[1]
        Tok = batch_embedding(x, self.E)
        X = Tok + self.P[:t]
        for i in range(nb_block):
            N1 = RMSNorm(X, self.blocks[i]["gamma1"])
            A = attention(N1, self.blocks[i]["W_Q"], self.blocks[i]["W_K"], self.blocks[i]["W_V"], self.blocks[i]["W_O"])
            X = X + A
            N2 = RMSNorm(X, self.blocks[i]["gamma2"])
            M = mlp(N2, self.blocks[i]["W1"], self.blocks[i]["W2"], self.blocks[i]["b1"], self.blocks[i]["b2"])
            X = X + M
        X_f = RMSNorm(X, self.gamma_final)

        logits = X_f @ self.W_out
        return logits

