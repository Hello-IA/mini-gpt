from collections import Counter, defaultdict
from pathlib import Path
from minigpt.tokenizer import BPETokenizer

K = 6000
CLEAN = Path("data/clean")
EXCLUS = {"hugo_Quatrevingt_Treize.txt", "hugo_Notre_Dame_de_Paris.txt"}   # val et test

tok = BPETokenizer()

# 1) fréquence de chaque chunk (mot, ponctuation, espaces...) sur le TRAIN uniquement
compte = Counter()
for f in sorted(CLEAN.glob("*.txt")):
    if f.name in EXCLUS:
        continue
    compte.update(tok.formatage(f.read_text(encoding="utf-8")))
print(len(compte), "chunks distincts")

# 2) BPE sur les chunks distincts, pondérés par leur fréquence
mots = [list(w) for w in compte]          # chaque chunk = liste d'octets (0..255)
freq = list(compte.values())
paires = Counter()
index = defaultdict(set)                  # paire -> indices des mots qui la contiennent
for i, m in enumerate(mots):
    for p in zip(m, m[1:]):
        paires[p] += freq[i]
        index[p].add(i)

regel = []
for k in range(K):
    if not paires:
        break
    meilleure = max(paires, key=paires.get)
    regel.append(meilleure)
    z = 256 + k                           # id du nouveau token
    for i in index.pop(meilleure):
        m = mots[i]
        for p in zip(m, m[1:]):           # retire les anciennes paires de ce mot
            paires[p] -= freq[i]
            if paires[p] <= 0:
                del paires[p]
        m = tok.merge(m, meilleure, z)
        mots[i] = m
        for p in zip(m, m[1:]):           # ajoute les nouvelles
            paires[p] += freq[i]
            index[p].add(i)
    paires.pop(meilleure, None)
    if k % 500 == 0:
        print("fusion", k, "/", K, flush=True)

# 3) sauvegarde (même format qu'avant)
tok.K = len(regel)
tok.regel = regel
tok.regel_deplis = []
tok.deplis_regel()
tok.writh("data/save.json")               # fais d'abord : cp data/save.json data/save_hugo.json

# 4) test aller-retour + compression sur le livre de validation
txt = (CLEAN / "hugo_Quatrevingt_Treize.txt").read_text(encoding="utf-8")
ids = [t for c in tok.encoder(txt) for t in c]
assert tok.decode([ids]) == txt, "aller-retour cassé"
print(f"{len(txt.encode('utf-8')) / len(ids):.2f} octets par token sur la val")
