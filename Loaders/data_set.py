import ipdb
import torch as t
import numpy as np
from torch.utils.data import Dataset, DataLoader

from configs import DefaultConfig
import os
import json
import itertools
from Predictor.Tools.DataTools.pad import pad




class MSDataSet(Dataset):
    def __init__(self, set):
        super(MSDataSet, self).__init__()
        self.file = os.path.join(DefaultConfig.processed_folder, set + '.json')
        with open(self.file) as reader:
            self.data = reader.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        line = json.loads(self.data[item])
        question_word = line['question']['text']
        question_char = [list(pad(word, DefaultConfig.word_max_lenth)) for word in line['question']['char']]
        passage_word = [passage['text'] for index, passage in line['passages'].items()]
        passage_char = [pad(passage['char'], DefaultConfig.word_max_lenth) for index, passage in line['passages'].items()]
        start = line['golden_span']['start']
        end = line['golden_span']['end']
        return question_word, question_char, passage_word, passage_char, start, end

def bucket_collect_func(batch):
    question_word, question_char, passage_word, passage_char, start, end = zip(*batch)
    ipdb.set_trace()


dataset = MSDataSet('dev')
dataloader = DataLoader(dataset, 4, True, collate_fn=bucket_collect_func)
for i in dataloader:
    print(i)

#
# def own_collate_fn(batch):
#
#     #pad batch
#     qwids = list(itertools.zip_longest(*qwids, fillvalue=pidx))
#     qwids = np.asarray(qwids).transpose().tolist()
#     cwids = list(itertools.zip_longest(*cwids, fillvalue=pidx))
#     cwids = np.asarray(cwids).transpose().tolist()
#     return t.LongTensor(qwids), t.LongTensor(cwids), t.LongTensor(baidx), t.LongTensor(eaidx)
#
#
#
#
# import ipdb
# import torch as t
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# class TestSet(Dataset):
#
#     def __init__(self, set):
#         super(TestSet, self).__init__()
#         self.datas = [[1,2],[1,2,3], [1,2,3,4],[1,2,3,4,5]]
#
#     def __len__(self):
#         return 100
#
#     def __getitem__(self, item):
#         return [2, 34, 5], [[1,2,3],[2,3,4],[3,4,5]]
#
#
# def collect_fun(batch):
#     a, b = zip(*batch)
#     print('-')
#     print(a)
#     print('--')
#     print(b)
#     print('-------')
#     batch = batch
#
# dataset = TestSet('dev')
#
# dataloader = DataLoader(dataset, 2, True, collate_fn=collect_fun)
# for i in dataloader:
#     print(i)