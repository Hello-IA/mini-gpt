import torch

B = 32
T = 256
d = 128
d_k = 32
n_heads = d//d_k
K = 6000
V = K +257
nb_block = 4
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
