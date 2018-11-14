from .data_set import MSDataSet, bucket_collect_func
from torch.utils.data import DataLoader


def get_dataloader(set, batch_size, num_worker):
    dataset = MSDataSet(set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, collate_fn=bucket_collect_func)
    return dataloader




"""
import pickle as pk
from configs import DefaultConfig
import ipdb
from Loaders import get_dataloader

word_vocab = pk.load(open(DefaultConfig.word_vocab, 'rb'))

dl = get_dataloader('dev', 1, 1)
for i in dl:
    pw = i[1]
    _, pl = pw.shape
    batch_size = 1
    pw = pw.view(batch_size, 11, pl).contiguous().view(batch_size, 11*pl)
    start = i[-3]
    end = i[-2]
    index = i[-1]
    ans = pw[0][start:end].tolist()
    print([word_vocab.from_id_token(i) for i in ans])
    ipdb.set_trace()
"""
