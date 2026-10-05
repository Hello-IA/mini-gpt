import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def entrainer_hf_bpe(path_train, vocab_size, save_path="data/hf_tokenizer.json"):
    if os.path.exists(save_path):
        return Tokenizer.from_file(save_path)

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size + 256)
    tok.train([path_train], trainer)

    tok.save(save_path) 
    return tok

entrainer_hf_bpe("data/corpus_train.txt", 6000)
