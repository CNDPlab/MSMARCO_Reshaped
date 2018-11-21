from Predictor.Tools.DataTools import build_data_structure
from Predictor.Tools.DataTools import tokenize_instance
from Predictor.Tools.DataTools import extract_golden_span
import json
from configs import DefaultConfig
import pickle as pk
from Predictor.Tools.Vocabs import VocabCollector


DATASETS = ['dev', 'train', 'eval']
word_vocab = pk.load(open(DefaultConfig.word_vocab, 'rb'))
char_vocab = pk.load(open(DefaultConfig.char_vocab, 'rb'))
vocab = VocabCollector(word_vocab, char_vocab)


def process_instance(raw_line):
    line = json.loads(raw_line)
    instance = build_data_structure(line)
    instance = tokenize_instance(instance)
    instance = extract_golden_span(instance)
    return instance


def convert2id(raw_line):
    instance = json.loads(raw_line)
    instance = vocab.transfer(instance)
    return instance



