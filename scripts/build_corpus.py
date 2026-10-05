import os

corpus_dir = "data"
files = sorted(f for f in os.listdir(corpus_dir) if f.startswith("hugo_") and f.endswith(".txt"))

# Choisis 2 livres/tomes distincts, de taille comparable si possible,
# un pour validation, un pour test — jamais les mêmes que ceux d'entraînement
val_files  = ["hugo_Quatrevingt_Treize.txt"]        # ~725 Ko
test_files = ["hugo_Notre_Dame_de_Paris.txt"]       # ~517 Ko

train_files = [f for f in files if f not in val_files + test_files]

def concat(file_list, output_name):
    with open(output_name, "w", encoding="utf-8") as out:
        for f in file_list:
            with open(os.path.join(corpus_dir, f), encoding="utf-8") as inp:
                out.write(inp.read())
                out.write("\n")
    print(f"{output_name}: {os.path.getsize(output_name)/1e6:.2f} Mo")

concat(train_files, "data/corpus_train.txt")
concat(val_files, "data/corpus_val.txt")
concat(test_files, "data/corpus_test.txt")
