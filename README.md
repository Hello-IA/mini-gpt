# mini-gpt

Un GPT construit from scratch (tokenizer BPE, attention, transformer, LoRA, GRPO), pour comprendre en profondeur les mathématiques derrière les LLM plutôt que de simplement les utiliser via une bibliothèque.

## Pourquoi ce projet

Après un stage en robotique/RL à l'ISIR (Sorbonne Université), je me suis rendu compte que la quasi-totalité des offres d'emploi ML/IA demandent des compétences LLM. Plutôt que de me contenter d'utiliser des bibliothèques haut niveau, j'ai voulu réimplémenter chaque brique moi-même — du tokenizer à l'entraînement par RL — pour être capable d'expliquer et de justifier chaque choix en entretien technique.

## État du projet

- [x] Tokenizer BPE (byte-level, from scratch)
- [ ] Attention (Q, K, V, masque causal, multi-têtes)
- [ ] Transformer complet + entraînement
- [ ] LoRA
- [ ] GRPO / RL

## Tokenizer BPE

Implémentation byte-level de Byte Pair Encoding, dans l'esprit du tokenizer de GPT-2 :

- Encodage en octets UTF-8 (alphabet de départ Σ = {0, ..., 255}), donc aucun caractère hors vocabulaire possible.
- Prétokenisation par expression régulière (module `regex`, gestion Unicode `\p{L}`), pour éviter que les fusions ne traversent certaines frontières (fin de mot / ponctuation).
- Entraînement glouton : à chaque étape, fusion de la paire d'octets adjacents la plus fréquente sur tout le corpus.
- Encodage et décodage testés avec `decode(encode(texte)) == texte`.

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Utilisation

```python
from minigpt.tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(text, vocab_size=1000)

ids = tok.encode("Le petit prince regardait les étoiles.")
texte = tok.decode(ids)
assert texte == "Le petit prince regardait les étoiles."
```

### Comparaison avec un tokenizer de référence (HuggingFace `tokenizers`)

Pour valider l'implémentation, le tokenizer BPE perso est comparé à un tokenizer BPE entraîné avec la bibliothèque `tokenizers` (HuggingFace), **à conditions égales** : même corpus d'entraînement (`corpus_train.txt`), même taille de vocabulaire (6256 tokens appris, hors octets de base), et évaluation sur le même texte jamais vu à l'entraînement (`corpus_val.txt`, un livre complet mis de côté).

| Métrique | BPE perso | HF `tokenizers` |
|---|---|---|
| Taux de compression (chars/token) | 3.19 | 3.16 |
| Fertilité, mots seuls (tokens/mot) | 1.39 | 1.38 |
| Couverture du vocabulaire | 89.3% | 91.0% |

**Lecture des résultats** : les trois métriques sont quasiment identiques entre les deux implémentations, ce qui valide que l'algorithme de fusion gloutonne BPE est correctement implémenté. La fertilité a été vérifiée mot par mot (voir `scripts/compare_tokenizers.py`) : les deux tokenizers identifient 99.996% des mêmes mots aux mêmes positions dans le texte, ce qui garantit que la comparaison porte bien sur la qualité des fusions apprises, et non sur un artefact de découpage. Le léger écart de couverture du vocabulaire s'explique par le corpus multi-livres : certains tokens appris (noms propres notamment) sont spécifiques à des livres absents de la validation.

*(Méthodologie détaillée des métriques : `scripts/compare_tokenizers.py`)*
