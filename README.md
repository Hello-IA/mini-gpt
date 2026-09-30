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

