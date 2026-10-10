# mini-gpt

Un GPT écrit **entièrement from scratch en PyTorch** : tokenizer BPE niveau octet, attention multi-têtes, transformer décoder, boucle d'entraînement. LoRA et GRPO suivront.

## Pourquoi ce projet

Comprendre les mathématiques des LLM en les implémentant moi-même, sans `nn.Linear` ni `nn.MultiheadAttention` : chaque formule (attention, normalisation, perte, gradient) est codée à la main et vérifiée par des tests.

## État du projet

- [x] Tokenizer BPE niveau octet
- [x] Pipeline de données (binaires `uint16`, `get_batch`)
- [x] Attention multi-têtes causale
- [x] Transformer (pre-norm, RMSNorm, MLP GELU) + entraînement
- [ ] Boucle de validation + sauvegarde du meilleur checkpoint
- [ ] Génération de texte (température, top-k, arrêt sur `<|eot|>`)
- [ ] LoRA
- [ ] GRPO

## Tokenizer BPE

Tokenizer BPE de type GPT-2, niveau octet : 256 octets de base + K = 6000 fusions = 6256 tokens, plus le token spécial `<|eot|>` (ID 6256).
**Taille du vocabulaire : V = 6257.**


| Métrique | BPE perso | HF `tokenizers` |
|---|---|---|
| Taux de compression (chars/token) | 3.19 | 3.16 |
| Fertilité, mots seuls (tokens/mot) | 1.39 | 1.38 |
| Couverture du vocabulaire | 89.3% | 91.0% |
## Corpus et pipeline de données

**Corpus** : romans de Victor Hugo (`data/clean/hugo_*.txt`), découpés par œuvre pour éviter toute fuite entre les ensembles.

| Ensemble | Œuvre(s) | Tokens |
|---|---|---|
| train | toutes les autres œuvres `hugo_*.txt` | 4 413 562 |
| val | *Quatrevingt-treize* | 431 129 |
| test | *Notre-Dame de Paris* | 306 259 |

- Chaque œuvre est tokenisée puis terminée par `<|eot|>` (ID 6256), écrit comme **entier**, pas comme texte.
- Stockage : tableau plat `uint16` (`tofile`), relu avec `np.fromfile(..., dtype=np.uint16).astype(np.int64)`.
- `data/meta.json` : `{"K": 6000, "N_train": 4413562, "N_val": 431129, "N_test": 306259}`.
- Le script de préparation vérifie la cohérence du fichier (relecture identique, taille = 2 × nombre de tokens, tous les IDs ≤ 6256).

**`get_batch(text, B, T, device)`** tire B positions de départ aléatoires `i` et renvoie :

- `x = text[i : i+T]`
- `y = text[i+1 : i+T+1]` (décalé d'un cran : le modèle prédit le token suivant à chaque position)

Cela donne T prédictions par séquence, soit B·T par batch. Le décalage de `y` est vérifié par un test (`x[:, 1:] == y[:, :-1]`).

## Architecture du modèle

Transformer décodeur, pre-norm.

| Symbole | Signification |
|---|---|
| B | taille du batch (séquences par pas) |
| T | longueur de contexte (tokens par séquence) |
| d | dimension du modèle |
| h | nombre de têtes d'attention |
| d_k = d / h | dimension par tête |
| d_ff | dimension cachée du MLP |
| V | taille du vocabulaire |
| E ∈ ℝ^{V×d} | matrice d'embedding des tokens |
| P ∈ ℝ^{T×d} | embedding de position (appris) |

**Forward**

1. X₀ = E[x] + P
2. Pour chaque bloc :
   - X ← X + Attention(RMSNorm(X; γ₁))
   - X ← X + MLP(RMSNorm(X; γ₂))
3. X_f = RMSNorm(X; γ_final)
4. logits = X_f · W_out, avec W_out ∈ ℝ^{d×V}

**Attention causale** : softmax(Q Kᵀ / √d_k + masque) V, avec masque = −∞ au-dessus de la diagonale, puis projection W_O.

**MLP** : GELU(X W₁ + b₁) W₂ + b₂.

**RMSNorm** : x / √(mean(x²) + ε) · γ.

**Perte** : entropie croisée moyenne sur les B·T positions. Le gradient par rapport aux logits vaut (p − onehot) / (B·T).

**Configuration actuelle**

| Hyperparamètre | Valeur |
|---|---|
| B | 32 |
| T | 256 |
| d | 128 |
| h | 4 |
| Blocs | 4 |
| d_ff | 512 |
| V | 6257 |
| Paramètres | 2 424 704 |
| Optimiseur | Adam, lr = 1e-3 |
| Précision | fp32 |

## Entraînement

Chaque pas utilise B·T = 8192 tokens. Une époque correspond à environ 539 pas, soit 4 413 562 / 8192.

Perte initiale attendue ≈ ln V ≈ 8,74 (observée : 8,76).

| Étape | Perte |
|---|---|
| Initialisation | 8,76 |
| Après 5000 pas (≈ 9,3 époques) | 3,13 (perplexité ≈ 22) |

La perte est ici mesurée sur le train. La validation reste à mettre en place pour détecter le surapprentissage.

## Vérifications de correction

| Test | Résultat |
|---|---|
| Perte initiale ≈ ln V | ✅ |
| Nombre de paramètres = formule théorique (2 424 704) | ✅ |
| Overfit d'un seul batch (perte → ~0) | ✅ (valide le flux de gradient, pas la causalité) |
| Causalité : modifier le token à la position 10 ne change pas les logits des positions < 10 | ✅ (écart = 0.0) |
| Décalage de y (`x[:,1:] == y[:,:-1]`) | ✅ |
| Tous les IDs < V dans les fichiers `.bin` | ✅ |
| Gradient de chaque paramètre non nul | à faire |

## Bugs rencontrés

1. **Paramètres non enregistrés** : les blocs étaient stockés dans une liste Python, donc invisibles pour `model.parameters()`. Corrigé avec `nn.ModuleList` de `nn.ParameterDict`.
2. **Opération in-place** : `X += A` casse l'autograd. Remplacé par `X = X + A`.
3. **Biais du MLP** : b₂ avait la taille d_ff au lieu de d.
4. **Corruption du fichier binaire** : `<|eot|>` avait été écrit comme 7 octets ASCII dans un fichier `uint16`, ce qui décalait tout et produisait des IDs hors vocabulaire. Corrigé en écrivant l'ID entier, avec des assertions de relecture.

## Prochaines étapes

1. Boucle de validation et sauvegarde du meilleur checkpoint.
2. Génération de texte (température, top-k, arrêt sur `<|eot|>`).
3. Passage à l'échelle : d = 256, 6 blocs, dropout, weight tying, AdamW avec warmup + cosine et gradient clipping.
4. LoRA, puis GRPO.
