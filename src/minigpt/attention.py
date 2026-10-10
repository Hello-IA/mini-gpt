from minigpt.tokenizer import BPETokenizer
from minigpt.batch import get_batch
from minigpt.embedding import init_embedding, batch_embedding
import torch
import math
from minigpt.rms_norm import RMSNorm
from minigpt.parametre import B, T, d, d_k, n_heads, K
from minigpt.init_matrice import init_Matrice

gamma = torch.ones((d,), device = "cuda", requires_grad = True) 
x, y = get_batch(torch.randint(0, K +256, (100000,), device = "cuda"), B, T,"cuda")
E = init_embedding(K + 256, d)
Tok = batch_embedding(x, E)
P = init_embedding(T, d)
X0 = Tok + P

W_Q = init_Matrice(d, d)
W_K = init_Matrice(d, d)
W_V = init_Matrice(d, d)
W_O = init_Matrice(d, d)

def reshape_mat(M):
    new_M = torch.reshape(M, (M.shape[0], M.shape[1], n_heads, d_k))
    new_M = torch.transpose(new_M, 1, 2)
    return new_M

def un_reshape_mat(M):
    new_M = torch.transpose(M, 1, 2)
    new_M = torch.reshape(new_M, (new_M.shape[0], new_M.shape[1], d))
    return new_M
def attention(h, W_Q, W_K, W_V, W_O):
    Q = h @ W_Q
    K = h @ W_K
    V = h @ W_V
    Q = reshape_mat(Q)
    K = reshape_mat(K)
    V = reshape_mat(V)
    K_transpose = torch.transpose(K, 2, 3)
    M = torch.full((T, T), float("-inf"), device = "cuda")
    M = torch.triu(M, diagonal = 1)

    S = (Q @ K_transpose/ math.sqrt(d_k)) + M

    A = torch.softmax(S, dim = -1)

    out = A@V
    concat = un_reshape_mat(out)
    return concat @ W_O


h_cp = X0.clone()
h_cp[:, 100] += 1
o1 = attention(X0, W_Q, W_K, W_V, W_O)
o2 = attention(h_cp, W_Q, W_K, W_V, W_O)
