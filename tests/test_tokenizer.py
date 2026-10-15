from pathlib import Path
from minigpt.tokenizer import BPETokenizer

textes = {
    "val (Hugo)": "data/clean/hugo_Quatrevingt_Treize.txt",
    "Germinal (Zola)": "data/clean/Germinal.txt",       # le nouveau tokenizer l'a vu à l'entraînement
}
for nom_tok in ["data/save_hugo.json", "data/save.json"]:
    tok = BPETokenizer.du_fichier(nom_tok)
    for nom, chemin in textes.items():
        txt = Path(chemin).read_text(encoding="utf-8")
        n = sum(len(c) for c in tok.encoder(txt))
        print(f"{nom_tok:22} {nom:18} {len(txt)/n:.2f} caractères/token  ({n:,} tokens)")
