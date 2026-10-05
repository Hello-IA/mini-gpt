from minigpt.tokenizer import BPETokenizer
import regex as re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers


bpe = BPETokenizer.du_fichier("data/save.json")

bpe_hf = Tokenizer.from_file("data/hf_tokenizer.json")

with open("data/corpus_val.txt", "r", encoding="utf-8") as f:
    text = f.read()


def taux_compretions(bpe, text):
    encode = bpe.encoder(text)
    tautale_token = 0
    for e in encode: 
        tautale_token += len(e)
    return len(text)/tautale_token

def fertilite_mots_seuls(bpe, text):

    total = 0
    nb_mots = 0

    sec_text_brut = bpe.formatage(text)
    encode = bpe.encoder(text)

    for sec_brut, sec_enc in zip(sec_text_brut, encode):
        text_sec = sec_brut.decode("utf-8").strip()
        if re.fullmatch(r"\p{L}+", text_sec):
            total += len(sec_enc)
            nb_mots += 1
    return total/nb_mots if nb_mots else None

def couverture(bpe, text):

    encode = bpe.encoder(text)
    encode = [token for list_tokens in encode for token in list_tokens]
    K = bpe.K +256
    tokens_utiliser = [0]*K
    for enc in encode:
        tokens_utiliser[enc] = 1
    nb_tokens = 0
    for t in tokens_utiliser:
        nb_tokens += t
    return nb_tokens/K

def taux_compretions_hf(bpe_hf, text):
    ids = bpe_hf.encode(text).ids
    return len(text)/len(ids)

def fertilite_mot_seul_hf(bpe_hf, text):
    encoding = bpe_hf.encode(text)
    word_ids = encoding.word_ids
    offsets = encoding.offsets

    tokens_par_mot = {}
    bornes_mot = {}

    for wid, (debut, fin) in zip(word_ids, offsets):
        if wid is None:
            continue
        tokens_par_mot[wid] = tokens_par_mot.get(wid, 0) + 1
        if wid not in bornes_mot:
            bornes_mot[wid] = [debut, fin]
        else:
            bornes_mot[wid][0] = min(bornes_mot[wid][0], debut)
            bornes_mot[wid][1] = max(bornes_mot[wid][1], fin)
    total = 0
    nb_mots = 0
    for wid, (debut, fin) in bornes_mot.items():
        mot = text[debut:fin].strip()
        if re.fullmatch(r"\p{L}+", mot):
            total += tokens_par_mot[wid]
            nb_mots += 1

    return total / nb_mots if nb_mots else None

def couverture_hf(bpe_hf, text):
    K = bpe_hf.get_vocab_size()
    ids = bpe_hf.encode(text).ids
    tokens_utilises = [0] * K
    for i in ids:
        tokens_utilises[i] = 1
    return sum(tokens_utilises) / K

print("taux de compretions :", taux_compretions(bpe, text))

print("frtilite :", fertilite_mots_seuls(bpe, text))


print("couverture du vocabulaire :", couverture(bpe, text))

print("taux de compretions hf :", taux_compretions_hf(bpe_hf, text))
print("fertilite hf :", fertilite_mot_seul_hf(bpe_hf, text))
print("couverture du vocabulaire hf:", couverture_hf(bpe_hf, text))
