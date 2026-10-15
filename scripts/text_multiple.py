from collections import Counter
from pathlib import Path

c = Counter()
for f in Path("data/clean").glob("*.txt"):
    lignes = {l.strip() for l in f.read_text(encoding="utf-8").split("\n") if 15 < len(l.strip()) < 200}
    c.update(lignes)
for l, n in c.most_common(40):
    if n >= 5:
        print(n, l[:120])
