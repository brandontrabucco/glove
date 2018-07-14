"""Author: Brandon Trabucco.
Very the installation of GloVe is function.
"""


import glove


config = glove.configuration.Configuration(
    embedding=50, 
    filedir="./embeddings/", 
    length=127,
    start_word="</StArT/>",
    end_word="</StoP/>",
    unk_word="</UnKnOwN/>")
vocab, embeddings = glove.load(config)


assert len(vocab.reverse_vocab) == 127, ""
for w in vocab.reverse_vocab:
    assert w in vocab.vocab, ""
    
    
assert vocab.word_to_id(config.start_word) == vocab.start_id, ""
assert vocab.word_to_id(config.end_word) == vocab.end_id, ""
assert vocab.word_to_id(config.unk_word) == vocab.unk_id, ""
assert vocab.word_to_id("./.2!#&*@^@%") == vocab.unk_id, ""


assert vocab.id_to_word(vocab.start_id) == config.start_word, ""
assert vocab.id_to_word(vocab.end_id) == config.end_word, ""
assert vocab.id_to_word(vocab.unk_id) == config.unk_word, ""
assert vocab.id_to_word(11182819) == config.unk_word, ""


assert embeddings.shape[0] == 127, ""
assert embeddings.shape[1] == 50, ""
assert embeddings.size == 127 * 50, ""


print("All test cases passed.")