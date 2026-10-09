from minigpt.tokenizer import BPETokenizer
import json

data = {}

bpe = BPETokenizer.du_fichier("data/save.json")

data["K"] = bpe.K

with open("data/s_train.bin", "rb") as f:
    data["N_train"] = len(f.read())

with open("data/s_val.bin", "rb") as f:
    data["N_val"] = len(f.read())
with open("data/s_test.bin", "rb") as f:
    data["N_test"] = len(f.read())

with open("data/meta.json", "w", encoding = "utf-8") as f:
    json.dump(data, f)
