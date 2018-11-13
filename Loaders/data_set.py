import ipdb
import torch as t
import numpy as np
from torch.utils.data import Dataset, DataLoader

from configs import DefaultConfig
import os
import json
import itertools
from tqdm import tqdm
from Predictor.Tools.DataTools.pad import pad


class MSDataSet(Dataset):
    def __init__(self, set):
        super(MSDataSet, self).__init__()
        self.file = os.path.join(DefaultConfig.processed_folder, set + '.json')
        with open(self.file) as reader:
            if set == 'dev':
                self.data = reader.readlines()[:1024]
            else:
                self.data = reader.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        line = json.loads(self.data[item])
        line['passages'] = sorted(line['passages'].items(), key=lambda x: int(x[0]), reverse=False)
        question_word = line['question']['text']
        question_char = [list(pad(word, DefaultConfig.word_max_lenth)) for word in line['question']['char']]
        passage_word = [passage['text'] for index, passage in line['passages']]
        passage_char = [[list(pad(word, DefaultConfig.word_max_lenth)) for word in passage['char']] for index, passage in line['passages']]
        start = line['golden_span']['start']
        end = line['golden_span']['end']
        passage_index = line['golden_span']['passage_index']
        return [question_word]*11, passage_word, [question_char]*11, passage_char, start, end, passage_index

def bucket_collect_func(batch):
    question_word, passage_word, question_char, passage_char, start, end, passage_index = zip(*batch)
    question_word = list(itertools.chain(*question_word))
    passage_word = list(itertools.chain(*passage_word))
    question_char = list(itertools.chain(*question_char))
    passage_char = list(itertools.chain(*passage_char))
    pad_char = [0] * DefaultConfig.word_max_lenth
    question_word = np.asarray(list(itertools.zip_longest(*question_word, fillvalue=0))).transpose()
    passage_word = np.asarray(list(itertools.zip_longest(*passage_word, fillvalue=0))).transpose()
    question_char = np.asarray(list(itertools.zip_longest(*question_char, fillvalue=pad_char))).transpose((1, 0, 2))
    passage_char = np.asarray(list(itertools.zip_longest(*passage_char, fillvalue=pad_char))).transpose((1, 0, 2))
    pad_lenth = passage_word.shape[1]
    try:
        start = np.array(start) + np.array(passage_index) * pad_lenth
        end = np.array(end) + np.array(passage_index) * pad_lenth
    except:
        ipdb.set_trace()
    return t.LongTensor(question_word), t.LongTensor(passage_word), t.LongTensor(question_char),\
           t.LongTensor(passage_char), t.LongTensor(start), t.LongTensor(end), t.LongTensor(np.asarray(passage_index))




if __name__ == '__main__':
    dataset = MSDataSet('dev')
    dataloader = DataLoader(dataset, 32, True, collate_fn=bucket_collect_func, num_workers=20)
    for i in tqdm(dataloader):
        ipdb.set_trace()
