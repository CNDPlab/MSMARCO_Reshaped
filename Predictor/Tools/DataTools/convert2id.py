from Predictor.Tools.DataTools import build_data_structure
from Predictor.Tools.DataTools import tokenize
from Predictor.Tools.DataTools import extract_golden_span
import json
from configs import DefaultConfig
import pickle as pk
from Predictor.Tools.Vocabs import VocabCollector


word_vocab = pk.load(open(DefaultConfig.word_vocab, 'rb'))
char_vocab = pk.load(open(DefaultConfig.char_vocab, 'rb'))
vocab = VocabCollector(word_vocab, char_vocab)
print('vocab loaded')


def convert2id(raw_line):
    instance = json.loads(raw_line)
    instance = vocab.transfer(instance)
    return instance