import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import numpy as np

import pickle
import os
import math

from features.tree_to_treeLSTM_input import convert_one_record, merge_into_batch


class TreeDataset(IterableDataset):

    def __init__(self, data_path, job_type, batch_size, fp_size=2048,
                 shuffle=False, device=torch.device('cpu'),
                 output_tree=False, to_tensor=False):
        '''

        :param file_list: a list of files
        '''
        super(TreeDataset).__init__()
        self.output_tree = output_tree  # whether output the orignal tree data
        # whether directly output sensor or convert to tensor later
        # don't convert to tensor when doing training, not compatible with parallel dataloader
        self.to_tensor = to_tensor
        self.data_path = data_path
        self.job_type = job_type
        self.batch_size = batch_size
        self.data_files = np.array(
            [os.path.join(self.data_path, x) for x in os.listdir(self.data_path) if self.job_type in x])
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.data_files)
        self.fp_size = fp_size
        self.shuffle = shuffle
        self.device = device

        self.merge_batch = lambda batch_inputs: merge_into_batch(batch_inputs,
                                                                 to_tensor=self.to_tensor,
                                                                 device=self.device)

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
        batch_trees = []
        for file in files:
            with open(file, 'rb') as fh:
                while True:
                    try:
                        one_record = pickle.load(fh)
                    except EOFError as _:
                        print(file)
                        if batch_inputs:
                            if self.output_tree:
                                yield batch_trees, self.merge_batch(batch_inputs)
                            else:
                                yield self.merge_batch(batch_inputs)
                        break
                    else:
                        batch_inputs.append(convert_one_record(one_record, fpsize=self.fp_size))
                        if self.output_tree:
                            batch_trees.append(one_record)

                    if len(batch_inputs) == self.batch_size:
                        if self.output_tree:
                            yield batch_trees, self.merge_batch(batch_inputs)
                        else:
                            yield self.merge_batch(batch_inputs)
                        batch_inputs = []
                        batch_trees = []


def get_tree_dataloader(data_path, job_type='train', fp_size=2048, device=torch.device('cpu'),
                        batch_size=4, num_workers=0, shuffle=False):
    tree_dataset = TreeDataset(data_path, job_type, batch_size=batch_size,
                               fp_size=fp_size, device=device, shuffle=shuffle)
    tree_dataloader = DataLoader(tree_dataset, batch_size=None, num_workers=num_workers)

    return tree_dataloader


if __name__ == '__main__':

    data_path = '/mnt/data/home/yiming/Projects/data/pathway_ranking/curated_data'
    tree_dataloader = get_tree_dataloader(data_path, job_type='train_198', fp_size=2048,
                                          device=torch.device('cpu'),
                                          batch_size=32, num_workers=0, shuffle=False)

    device = torch.device('cpu')
    fp_size = 2048
    lstm_size = 512
    for batch in tree_dataloader:
        print(batch['batch_size'])
        pass
