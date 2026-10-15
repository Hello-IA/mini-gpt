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

