import json
import numpy as np
from minigpt.tokenizer import BPETokenizer

bpe = BPETokenizer.du_fichier("data/save.json")

def nb_tokens(path):
    return len(np.fromfile(path, dtype=np.uint16))   # 2 octets par token

data = {
    "K": bpe.K,
    "V": 256 + bpe.K + 1,                  # 256 octets + K fusions + <|eot|>
    "N_train": nb_tokens("data/s_train.bin"),
    "N_val": nb_tokens("data/s_val.bin"),
    "N_test": nb_tokens("data/s_test.bin"),
}

with open("data/meta.json", "w", encoding="utf-8") as f:
    json.dump(data, f)
print(data)
