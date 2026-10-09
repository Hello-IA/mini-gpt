import os
from minigpt.tokenizer import BPETokenizer
import numpy as np


bpe = BPETokenizer.du_fichier("data/save.json")
corpus_dir = "data/clean"
files = sorted(f for f in os.listdir(corpus_dir) if f.startswith("hugo_") and f.endswith(".txt"))

val_files  = ["hugo_Quatrevingt_Treize.txt"]    
test_files = ["hugo_Notre_Dame_de_Paris.txt"]  

train_files = [f for f in files if f not in val_files + test_files]

def concat(file_list, output_name):
    with open(output_name, "wb") as out:
        for f in file_list:
            
            with open(os.path.join(corpus_dir, f), encoding="utf-8") as inp:
                encode = bpe.encoder(inp.read())
                encode = [tok for most in encode for tok in most]
                encode = np.array(encode, dtype = np.int16)
                out.write(encode)
                out.write(b"<|eot|>")
    print(f"{output_name}: {os.path.getsize(output_name)/1e6:.2f} Mo")

concat(train_files, "data/s_train.bin")
concat(val_files, "data/s_val.bin")
concat(test_files, "data/s_test.bin")

