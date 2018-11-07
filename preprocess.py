from Predictor.Tools.DataTools.main import process_instance
from Predictor.Tools.Iters import process_file_mul
import os
from configs import DefaultConfig
from Predictor.Tools.Vocabs import Vocab
from Predictor.Tools.Iters import MiddleIter
import json
from tqdm import tqdm
import gensim




files = ['train', 'dev']


for file in files:
    process_file_mul(
        os.path.join(DefaultConfig.raw_folder, file+'.json'),
        process_instance,
        os.path.join(DefaultConfig.middle_folder, file+'.json'),
        20
    )

# Build word and char vocab
model = gensim.models.KeyedVectors.load_word2vec_format(DefaultConfig.gensim_file, binary=False)  # GloVe Model
word_vocab = Vocab()
char_vocab = Vocab()

MiddleIter = MiddleIter('train')
for line in tqdm(MiddleIter):
    line = json.loads(line)
    for index, answer in line['answers'].items():
        if answer['text'] != '':
            word_vocab.add_sentance(answer['text'])
        for word in answer['char']:
            char_vocab.add_sentance(word)
    for index, passage in line['passages'].items():
        if passage['text'] != '':
            word_vocab.add_sentance(passage['text'])
        for word in passage['char']:
            char_vocab.add_sentance(word)

    word_vocab.add_sentance(line['question']['text'])
    for word in line['question']['char']:
        char_vocab.add_sentance(word)

word_vocab.filter_rare_word_build_vocab(10)
word_vocab.use_pretrained(model)
word_vocab.save(DefaultConfig.word_vocab)

char_vocab.filter_rare_word_build_vocab(300)
char_vocab.save(DefaultConfig.char_vocab)

from Predictor.Tools.DataTools.convert2id import convert2id
# convert

for file in files:
    process_file_mul(
        os.path.join(DefaultConfig.middle_folder, file+'.json'),
        convert2id,
        os.path.join(DefaultConfig.processed_folder, file+'.json'),
        20
    )
