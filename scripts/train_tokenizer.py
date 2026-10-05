from minigpt.tokenizer import BPETokenizer

bpe = BPETokenizer.du_texte("data/corpus_train.txt", K=6000)
bpe.BPE()
bpe.deplis_regel()
bpe.writh("data/save.json")
