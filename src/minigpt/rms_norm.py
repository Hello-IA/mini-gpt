from minigpt.tokenizer import BPETokenizer
from minigpt.batch import get_batch
import torch



def RMSNorm(X, gamma):
    ms = torch.mean(X**2, dim = -1, keepdim=True)
    return gamma*X/torch.sqrt(ms + 1e-5)
