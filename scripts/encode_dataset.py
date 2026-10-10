import os
from minigpt.tokenizer import BPETokenizer
import numpy as np


bpe = BPETokenizer.du_fichier("data/save.json")
corpus_dir = "data/clean"
files = sorted(f for f in os.listdir(corpus_dir) if f.startswith("hugo_") and f.endswith(".txt"))

val_files  = ["hugo_Quatrevingt_Treize.txt"]    
test_files = ["hugo_Notre_Dame_de_Paris.txt"]  

train_files = [f for f in files if f not in val_files + test_files]

EOT_ID = 6256   # premier ID libre : le vocabulaire BPE occupe 0..6255

def concat(file_list, output_name):
    parts = []
    for f in file_list:
        with open(os.path.join(corpus_dir, f), encoding="utf-8") as inp:
            ids = [tok for chunk in bpe.encoder(inp.read()) for tok in chunk]
        parts.append(np.array(ids + [EOT_ID], dtype=np.int64))   # séparateur = un entier

    arr = np.concatenate(parts)
    assert arr.min() >= 0 and arr.max() <= EOT_ID
    arr.astype(np.uint16).tofile(output_name)

    back = np.fromfile(output_name, dtype=np.uint16)              # relecture de contrôle
    assert len(back) == len(arr)
    assert os.path.getsize(output_name) == 2 * len(arr)
    assert (back == arr).all()
    print(f"{output_name}: {len(arr)} tokens, {os.path.getsize(output_name)/1e6:.2f} Mo")
    return len(arr)
concat(train_files, "data/s_train.bin")
concat(val_files, "data/s_val.bin")
concat(test_files, "data/s_test.bin")

