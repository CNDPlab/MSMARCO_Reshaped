import pickle as pk
from configs import DefaultConfig


def load_vocab():
    return pk.load(open(DefaultConfig.word_vocab, 'rb'))
