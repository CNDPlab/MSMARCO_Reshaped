from .data_set import MSDataSet, bucket_collect_func
from torch.utils.data import DataLoader


def get_dataloader(set, batch_size, num_worker):
    dataset = MSDataSet(set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker,collate_fn=bucket_collect_func)
    return dataloader
