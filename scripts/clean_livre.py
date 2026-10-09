# scripts/build_corpus.py
from pathlib import Path

DATA_DIR = Path("data")

OUTPUT = DATA_DIR / "corpus_complet.txt"

START_MARKER = "*** START OF"
END_MARKER = "*** END OF"


fichiers = sorted(DATA_DIR.glob("hugo_*.txt"))

for f in fichiers:
    texte = f.read_text(encoding="utf-8")
    debut = texte.find(START_MARKER)
    fin = texte.find(END_MARKER)

    if debut == -1 or fin == -1:
        print(f"ATTENTION: marqueurs non trouvés dans {f.name}, fichier ignoré")
        continue

    # saute la ligne du marqueur de début elle-même (jusqu'au prochain retour à la ligne)
    debut_contenu = texte.find("\n", debut) + 1
    contenu = texte[debut_contenu:fin].strip()
    (Path("data/clean")/f.name).write_text(contenu, encoding="utf-8")
    print(f"{f.name}: {len(contenu):,} caractères gardés")

