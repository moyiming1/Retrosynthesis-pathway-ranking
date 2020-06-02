import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import numpy as np

from itertools import cycle
import pickle
import os
import math

from data_prep.utils import convert_one_record_to_ds, merge_ds_into_batch


class DsDataset(IterableDataset):

    def __init__(self, data_path, job_type, batch_size,
                 shuffle=False, device=torch.device('cpu'),
                 max_step=12, to_tensor=False):
        '''

        :param file_list: a list of files
        '''
        super(DsDataset).__init__()
        self.to_tensor = to_tensor
        self.data_path = data_path
        self.job_type = job_type
        self.batch_size = batch_size
        self.max_step = max_step
        self.data_files = np.array(
            [os.path.join(self.data_path, x) for x in os.listdir(self.data_path) if self.job_type in x])
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.data_files)

        self.shuffle = shuffle
        self.device = device

        self.merge_batch = lambda batch_inputs: merge_ds_into_batch(batch_inputs)

    def __iter__(self):
        '''

        :return:
        '''
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            files = self.data_files
        else:
            per_worker = int(math.ceil(len(self.data_files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.data_files))

            files = self.data_files[iter_start:iter_end]
        batch_inputs = []
        for file in files:
            with open(file, 'rb') as fh:
                while True:
                    try:
                        one_record = pickle.load(fh)
                    except EOFError as _:
                        print(file)
                        if batch_inputs:
                            yield self.merge_batch(batch_inputs)
                        break
                    else:
                        batch_inputs.append(convert_one_record_to_ds(one_record))

                    if len(batch_inputs) == self.batch_size:
                        yield self.merge_batch(batch_inputs)
                        batch_inputs = []


def get_ds_dataloader(data_path, job_type='train', max_step=12,  device=torch.device('cpu'),
                      batch_size=4, num_workers=0, shuffle=False):
    sc_dataset = DsDataset(data_path, job_type, batch_size=batch_size,
                           max_step=max_step, device=device, shuffle=shuffle)
    sc_dataloader = DataLoader(sc_dataset, batch_size=None, num_workers=num_workers)

    return sc_dataloader
