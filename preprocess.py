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
            word_vocab.add_sentance_token(answer['text'])
        for word in answer['char']:
            char_vocab.add_sentance_token(word)
    for index, passage in line['passages'].items():
        if passage['text'] != '':
            word_vocab.add_sentance_token(passage['text'])
        for word in passage['char']:
            char_vocab.add_sentance_token(word)

    word_vocab.add_sentance_token(line['question']['text'])
    for word in line['question']['char']:
        char_vocab.add_sentance_token(word)

word_vocab.filter_rare_token_build_vocab(5)
word_vocab.use_pretrained(model)
word_vocab.save(DefaultConfig.word_vocab)

char_vocab.filter_rare_token_build_vocab(300)
char_vocab.random_init(DefaultConfig.char_embedding_dim)
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


os.rename(
    os.path.join(DefaultConfig.processed_folder, 'train.json'),
    os.path.join(DefaultConfig.processed_folder, 'train_tmp.json')
)

# drop train :score < 0.8 and have_answer = True
count = 0
with open(os.path.join(DefaultConfig.processed_folder, 'train_tmp.json')) as reader:
    with open(os.path.join(DefaultConfig.processed_folder, 'train.json'), 'w') as writer:
        for i in tqdm(reader):
            line = json.loads(i)
            if line['golden_span']['score'] > 0.8 and line['golden_span']['end'] >= 0 and line['golden_span']['start'] >= 0:
                json.dump(line, writer)
                writer.write('\n')
            else:
                count += 1


print(f'{count} droped')
#############
os.rename(
    os.path.join(DefaultConfig.processed_folder, 'dev.json'),
    os.path.join(DefaultConfig.processed_folder, 'dev_tmp.json')
)

count = 0
with open(os.path.join(DefaultConfig.processed_folder, 'dev_tmp.json')) as reader:
    with open(os.path.join(DefaultConfig.processed_folder, 'dev.json'), 'w') as writer:
        for i in tqdm(reader):
            line = json.loads(i)
            if line['golden_span']['score'] > 0 and line['golden_span']['end'] >= 0 and line['golden_span']['start'] >= 0:
                json.dump(line, writer)
                writer.write('\n')
            else:
                count += 1


print(f'{count} droped')


