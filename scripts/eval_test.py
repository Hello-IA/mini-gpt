"""Évalue un checkpoint sur val et test (fenêtres non chevauchantes de T tokens).

Usage : PYTHONPATH=src python scripts/eval_test.py checkpoints/best_19M.pt
Le modèle est construit avec minigpt.parametre : il doit correspondre au checkpoint.
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from minigpt.parametre import V, d, T, nb_block, device
from minigpt.model import LLM_hugo

SPLITS = {
    "val  (Quatrevingt-treize)": ("data/s_val.bin", "data/clean/hugo_Quatrevingt_Treize.txt"),
    "test (Notre-Dame de Paris)": ("data/s_test.bin", "data/clean/hugo_Notre_Dame_de_Paris.txt"),
}
BATCH = 64


@torch.no_grad()
def evaluer(model, tokens):
    """Perte moyenne par token sur toutes les fenêtres de T tokens du split."""
    n_fen = (len(tokens) - 1) // T
    nll, n = 0.0, 0
    for a in range(0, n_fen, BATCH):
        starts = [int(k) * T for k in range(a, min(a + BATCH, n_fen))]
        x = torch.stack([tokens[i : i + T] for i in starts]).to(device)
        y = torch.stack([tokens[i + 1 : i + T + 1] for i in starts]).to(device)
        logits = model(x)  # (B, T, V)
        nll += F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    return nll / n, n


def main(chemin):
    ckpt = torch.load(chemin, map_location=device)
    attendu = {"V": V, "d": d, "T": T, "nb_block": nb_block}
    assert ckpt["config"] == attendu, f"checkpoint {ckpt['config']} != parametre.py {attendu}"

    model = LLM_hugo()
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{chemin} | pas {ckpt['step']} | {n_par:,} paramètres | val sauvegardée {ckpt['val_loss']:.4f}")

    for nom, (bin_path, txt_path) in SPLITS.items():
        tokens = torch.from_numpy(np.fromfile(bin_path, dtype=np.uint16).astype(np.int64))
        loss, n = evaluer(model, tokens)
        chars = len(open(txt_path, encoding="utf-8").read())
        cpt = chars / len(tokens)  # caractères par token mesurés sur CE texte
        bpc = loss / math.log(2) / cpt
        print(f"{nom:28s} | tokens {n:>8d} | perte {loss:.4f} | ppl {math.exp(loss):6.2f} "
              f"| {cpt:.2f} car/token | {bpc:.3f} bpc")
        if nom.startswith("val") and abs(loss - ckpt["val_loss"]) > 0.05:
            print("  ATTENTION : la perte val diffère de celle de l'entraînement de plus de 0,05. "
                  "Vérifie le nombre de têtes (h) dans parametre.py : à d fixé, un mauvais h "
                  "charge sans erreur mais donne un modèle faux.")


if __name__ == "__main__":
    main(sys.argv[1])
