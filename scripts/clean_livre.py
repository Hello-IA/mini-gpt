# scripts/clean_corpus.py
import re, statistics
from collections import Counter
from pathlib import Path

RAW, CLEAN = Path("data/raw"), Path("data/clean")
CLEAN.mkdir(parents=True, exist_ok=True)

UNWRAP = True        # recolle les lignes coupées à ~70 caractères (prose uniquement)
MIN_CHARS = 20_000   # alerte si un fichier nettoyé est plus court

START_RE = re.compile(r"\*{3}\s*START OF.*?\*{3}", re.S | re.I)
END_RE = re.compile(r"\*{3}\s*END OF", re.I)
# lignes de crédits à supprimer (Gutenberg / Wikisource)
BOILER_RE = re.compile(r"wikisource|wikimedia|wikip[ée]dia|creative commons|gutenberg", re.I)
LIGNES_RE = re.compile(
    r"^(?:ŒUVRES COMPLÈTES|ÉDITION DÉFINITIVE D['’]APRÈS LES MANUSCRITS ORIGINAUX|7, RUE PERRONET, 7"
    r"|TOUS DROITS RÉSERVÉS|TABLE DES MATIÈRES\.?|Table des matières|FIN DE LA TABLE\.?"
    r"|-{10,}|\(ne fait pas partie de l[’']ouvrage original\))$"
    r"|RUE DES GRANDS-AUGUSTINS, 5\.?|(?:\.\s?){5,}"
)
NOTE_RE = re.compile(
    r"Cette version numérisée reproduit|La ponctuation n['’]a pas été modifiée"
    r"|les corrections suivantes ont été apportées|Les défauts d['’]impression en début et en fin de ligne"
    r"|tacitement corrig|Liste des modifications", re.I)

def sans_notes(t):
    return "\n\n".join(b for b in t.split("\n\n") if not NOTE_RE.search(b))
# normalisations
REPLACE = {
    "\u2010": "-", "\u2011": "-", "\u2012": "—", "\u2015": "—",
    "ſ": "s", "ʼ": "’", "\u2028": "\n", "\u2029": "\n\n",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
}
# espaces exotiques -> espace normal
SPACES_RE = re.compile(r"[\t\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
# SEUL ce qui est autorisé reste : ASCII, lettres latines (accents), ponctuation typographique
NOT_ALLOWED = re.compile(
    "[^\n -~\u00A1-\u00AC\u00AE-\u00FF\u0152\u0153\u0178"
    "\u2013\u2014\u2018-\u201E\u2020-\u2022\u2026\u2030\u2039\u203A\u20AC]"
)


def recoller(t):
    """Recolle les paragraphes en prose coupés par des retours à la ligne fixes.
    Les strophes de vers (lignes courtes) ne sont pas touchées."""
    out, n = [], 0
    for bloc in t.split("\n\n"):
        lignes = [l.strip() for l in bloc.split("\n") if l.strip()]
        if len(lignes) >= 2 and statistics.median(len(l) for l in lignes[:-1]) >= 60:
            out.append(" ".join(lignes))
            n += 1
        else:
            out.append("\n".join(lignes))
    return "\n\n".join(out), n


def nettoyer(t, rapport):
    # 1) en-tête et pied de page Gutenberg
    m = START_RE.search(t)
    if m:
        t = t[m.end():]
    ends = list(END_RE.finditer(t))
    if ends:
        t = t[:ends[-1].start()]
    rapport["marqueurs"] = bool(m and ends)
    # colophon BEQ : on coupe de "À propos de cette édition électronique" jusqu'à la fin
    i = t.rfind("À propos de cette édition électronique")
    rapport["colophon_hors_fin"] = (i != -1 and i <= 0.8 * len(t))   # à vérifier à la main
    if i != -1 and i > 0.8 * len(t):
        t = t[:i]
    j = t.rfind("Liste des modifications")        # liste de corrections en fin de livre
    if j != -1 and j > 0.85 * len(t):
        t = t[:j]
    # table des matières encadrée par ses deux marqueurs
    t = re.sub(r"TABLE DES MATIÈRES\.?\s.{0,12000}?FIN DE LA TABLE\.?", "", t, flags=re.S)
    # note de transcription BEQ
    t = re.sub(r"Cette version numérisée reproduit.{0,500}?ont été apportées\.", "", t, flags=re.S)
    # 2) normalisation des caractères
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    for a, b in REPLACE.items():
        t = t.replace(a, b)
    t = SPACES_RE.sub(" ", t)
    # notes de bas de page Wikisource : lignes commençant par "↑"
    t = re.sub(r"(?m)^[ ]*↑.*\n?", "", t)
    # appels de notes en exposant (¹ ² ³ ⁴ ... ⁹)
    t = re.sub(r"[\u00b9\u00b2\u00b3\u2070-\u2079]+", "", t)
    rapport["supprimes"] = Counter(NOT_ALLOWED.findall(t))   # invisibles, BOM, symboles étranges...
    t = NOT_ALLOWED.sub("", t)

    # 3) tiret de dialogue unifié ("—", "–" en début de ligne -> "--")
    t = re.sub(r"(?m)^[ ]*[—–][ ]?", "--", t)

    # 4) lignes de crédits
    lignes = t.split("\n")
    gardees = [l for l in lignes if not BOILER_RE.search(l) and not LIGNES_RE.match(l.strip())]
    rapport["credits"] = len(lignes) - len(gardees)
    t = "\n".join(gardees)

    # 5) espaces et lignes vides
    t = re.sub(r"[ ]+\n", "\n", t)
    t = re.sub(r"\n[ ]+", "\n", t)
    t = re.sub(r"[ ]{2,}", " ", t)
    t = sans_notes(t)
    t = re.sub(r"(?m)^--[ ]+", "--", t)        # "-- Oui" -> "--Oui"
    t = re.sub(r"[ ]+([?!;:])", r"\1", t)      # "dire ?" -> "dire?"
    if UNWRAP:
        t, rapport["recolles"] = recoller(t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip() + "\n"


total = 0
bizarres = Counter()
alertes = []
for f in sorted(RAW.glob("*.txt")):
    brut = f.read_text(encoding="utf-8", errors="replace")
    rap = {}
    propre = nettoyer(brut, rap)
    (CLEAN / f.name).write_text(propre, encoding="utf-8")
    bizarres.update(rap["supprimes"])
    total += len(propre.encode("utf-8"))
    if not rap["marqueurs"]:
        alertes.append(f"{f.name}: pas de marqueurs Gutenberg (vérifie début/fin à la main)")
    if rap.get("colophon_hors_fin"):
        alertes.append(f"{f.name}: colophon BEQ trouvé hors de la fin, à vérifier à la main")
    if len(propre) < MIN_CHARS:
        alertes.append(f"{f.name}: très court ({len(propre):,} caractères), téléchargement incomplet ?")
    print(f"{f.name}: {len(propre):,} car. | crédits retirés {rap['credits']} | "
          f"paragraphes recollés {rap.get('recolles', 0)}")

print(f"\nTOTAL : {total/1e6:.1f} Mo dans {CLEAN}/")
print("\nCaractères supprimés (les plus fréquents) :")
for ch, n in bizarres.most_common(30):
    print(f"  {ch!r:12} U+{ord(ch):04X}  x{n}")
if alertes:
    print("\nÀ vérifier :")
    for a in alertes:
        print("  -", a)
