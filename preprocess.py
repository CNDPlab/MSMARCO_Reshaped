from Predictor.Tools.DataTools import process_instance
from Predictor.Tools.Iters import process_file, process_file_mul
import os
from configs import DefaultConfig
from Predictor.Tools.Vocabs import Vocab
from Predictor.Tools.Iters import MiddleIter
import json
import gensim




files = ['dev']


for file in files:
    process_file_mul(
        os.path.join(DefaultConfig.raw_folder, file+'.json'),
        process_instance,
        os.path.join(DefaultConfig.middle_folder, file+'.json'),
        20
    )


model = gensim.models.KeyedVectors.load_word2vec_format(DefaultConfig.gensim_file, binary=False)  # GloVe Model
vocab = Vocab()
MiddleIter = MiddleIter('train')
for line in MiddleIter:
    line = json.loads(line)
    for index, answer in line['answers'].items():
        if answer['text'] != '':
            vocab.add_sentance(answer['text'])
    for index, passage in line['passages'].items():
        if passage['text'] != '':
            vocab.add_sentance(passage['text'])
    vocab.add_sentance(line['question']['text'])

vocab.filter_rare_word_build_vocab(100)
vocab.use_pretrained(model)
vocab.save('Datas/vocab.pkl')