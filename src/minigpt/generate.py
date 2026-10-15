import torch
from minigpt.parametre import V, d, T, nb_block, device
from minigpt.model import LLM_hugo
from minigpt.tokenizer import BPETokenizer      # adapte l'import à ton arborescence

CKPT = "checkpoints/best_19M.pt"
EOT_ID = 6256

# --- modèle ---
ckpt = torch.load(CKPT, map_location=device)
assert ckpt["config"] == {"V": V, "d": d, "T": T, "nb_block": nb_block}, \
    f"parametre.py ne correspond pas au checkpoint : {ckpt['config']}"
model = LLM_hugo()
model.load_state_dict(ckpt["model"])
model.eval()
print(f"modèle chargé (pas {ckpt['step']}, val {ckpt['val_loss']:.3f})")

# --- tokenizer ---
bpe = BPETokenizer.du_fichier("data/save.json")


def encode(text):
    return [t for chunk in bpe.encoder(text) for t in chunk]


def decode(ids):
    return bpe.decode([ids])          # <- adapte au nom réel de ta méthode de décodage


@torch.no_grad()
def generate(prompt, max_new=300, temperature=0.8, top_k=40):
    ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new):
        ctx = ids[:, -T:]                                   # fenêtre glissante de T tokens
        logits = model(ctx)[:, -1, :] / max(temperature, 1e-4)   # logits du dernier token seulement
        if top_k is not None:
            kth = torch.topk(logits, top_k).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        if nxt.item() == EOT_ID:                            # fin d'œuvre : on s'arrête
            break
        ids = torch.cat([ids, nxt], dim=1)
    return decode(ids[0].tolist())


if __name__ == "__main__":
    print("Écris un début de texte (vide pour quitter).")
    print("Commandes : /t 0.7 (température), /k 20 (top-k), /n 200 (longueur)\n")
    temperature, top_k, max_new = 0.8, 40, 300
    while True:
        prompt = input("> ")
        if not prompt.strip():
            break
        if prompt.startswith("/t "):
            temperature = float(prompt[3:]); continue
        if prompt.startswith("/k "):
            top_k = int(prompt[3:]); continue
        if prompt.startswith("/n "):
            max_new = int(prompt[3:]); continue
        print(generate(prompt, max_new, temperature, top_k), "\n")
