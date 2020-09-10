"""
This module defines the PathwayRankingHandler for use in Torchserve.
"""

import os

import scipy.sparse
import torch
from ts.torch_handler.base_handler import BaseHandler


class PathwayRankingHandler(BaseHandler):

    def __init__(self):
        super().__init__()
        self.fp_size = 2048
        self.lstm_size = 512

    def initialize(self, context):
        properties = context.system_properties
        self.device = torch.device('cuda:' + str(properties.get('gpu_id')) if torch.cuda.is_available() else 'cpu')

        model_dir = properties.get('model_dir')
        model_pt_path = os.path.join(model_dir, "treeLSTM512-fp2048.pt")
        model_def_path = os.path.join(model_dir, "model.py")

        from model import PathwayRankingModel
        state = torch.load(model_pt_path, map_location=self.device)
        state_dict = state['state_dict']
        self.model = PathwayRankingModel(self.fp_size, self.lstm_size, encoder=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.initalized = True
        print('Model file {0} loaded successfully.'.format(model_pt_path))

    def preprocess(self, data):
        batch = data[0].get('data') or data[0].get('body')

        # Expand sparse matrices
        shape = (len(batch['node_order']), self.fp_size)
        pfp = scipy.sparse.csc_matrix(tuple(batch['pfp']), shape=shape).toarray()
        rxnfp = scipy.sparse.csc_matrix(tuple(batch['rxnfp']), shape=shape).toarray()

        batch['pfp'] = torch.tensor(pfp, device=self.device, dtype=torch.float32)
        batch['rxnfp'] = torch.tensor(rxnfp, device=self.device, dtype=torch.float32)
        batch['node_order'] = torch.tensor(batch['node_order'], device=self.device, dtype=torch.int64)
        batch['adjacency_list'] = torch.tensor(batch['adjacency_list'], device=self.device, dtype=torch.int64)
        batch['edge_order'] = torch.tensor(batch['edge_order'], device=self.device, dtype=torch.int64)

        return batch

    def inference(self, data, *args, **kwargs):
        pfp = data['pfp']
        rxnfp = data['rxnfp']
        adjacency_list = data['adjacency_list']
        node_order = data['node_order']
        edge_order = data['edge_order']
        num_nodes = data['num_nodes']

        # Forward pass
        scores, encoded_trees = self.model(pfp, rxnfp, adjacency_list, node_order, edge_order, num_nodes)

        return {'scores': scores, 'encoded_trees': encoded_trees}

    def postprocess(self, data):
        return [{
            'scores': data['scores'].view(-1,).tolist(),
            'encoded_trees': data['encoded_trees'].tolist(),
        }]
