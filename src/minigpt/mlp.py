from minigpt.rms_norm import RMSNorm
from minigpt.attention import attention
from minigpt.parametre import B, T, d, d_k, n_heads, K, device
from minigpt.init_matrice import init_Matrice
from minigpt.tokenizer import BPETokenizer
from minigpt.batch import get_batch
from minigpt.embedding import init_embedding, batch_embedding
import torch

def mlp(x, W1, W2, b1, b2):
    return torch.nn.functional.gelu(x@W1 + b1)@W2 + b2

gamma1 = torch.ones((d,), device = "cuda", requires_grad = True)
gamma2 = torch.ones((d,), device = "cuda", requires_grad = True)

x, y = get_batch(torch.randint(0, K +256, (100000,), device = device), B, T,device)
E = init_embedding(K + 256, d)
Tok = batch_embedding(x, E)
P = init_embedding(T, d)
X0 = Tok + P

W_Q = init_Matrice(d, d)
W_K = init_Matrice(d, d)
W_V = init_Matrice(d, d)
W_O = init_Matrice(d, d)

W1 = init_Matrice(d, 4*d)
W2 = init_Matrice(4*d, d)

b1 = torch.zeros((4*d,), device = "cuda", requires_grad = True)
b2 = torch.zeros((d,), device = "cuda", requires_grad = True)


h = RMSNorm(X0, gamma1)
a = attention(h, W_Q, W_K, W_V, W_O)
an = RMSNorm(a, gamma2)
o1 = mlp(an, W1, W2, b1, b2)
print(o1)
print(o1.shape)
