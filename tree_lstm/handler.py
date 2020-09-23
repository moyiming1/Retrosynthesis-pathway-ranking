"""
This module defines the PathwayRankingHandler for use in Torchserve.

Note that imports look odd because files end up in the same directory once
they are converted into a model archive and loaded into Torchserve.
"""

import os

import torch
from ts.torch_handler.base_handler import BaseHandler

from tree_to_treeLSTM_input import convert_tree_to_singleinput, merge_into_batch


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
        trees = data[0].get('data') or data[0].get('body')
        return merge_into_batch([convert_tree_to_singleinput(tree) for tree in trees],
                                to_tensor=True, device=self.device)

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
