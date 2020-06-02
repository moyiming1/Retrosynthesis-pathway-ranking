import os, sys

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

import torch
from tree_lstm.treeLSTM_model import PathwayRankingModel
from features.tree_to_treeLSTM_input import convert_tree_to_singleinput, merge_into_batch
import hdbscan
import sklearn.cluster as cluster
import numpy as np


class PathwayRanker():
    def __init__(self, device, fp_size=2048, lstm_size=512):
        self.tree_converter = convert_tree_to_singleinput
        self.device = device
        self.fp_size = fp_size
        self.model = PathwayRankingModel(fp_size, lstm_size, encoder=True).to(self.device)

    def load_weights(self, model_path):
        # Load model and args
        state = torch.load(model_path, map_location=lambda storage, loc: storage)
        args, loaded_state_dict = state['args'], state['state_dict']
        self.model.load_state_dict(loaded_state_dict)
        self.model.eval()

    def scorer(self, trees, clustering=False, cluster_method='hdbscan',
               min_samples=5, min_cluster_size=5):
        batch = merge_into_batch([convert_tree_to_singleinput(tree, fpsize=self.fp_size)
                                  for tree in trees], to_tensor=True, device=self.device)

        pfp = batch['pfp']
        rxnfp = batch['rxnfp']
        adjacency_list = batch['adjacency_list']
        node_order = batch['node_order']
        edge_order = batch['edge_order']
        num_nodes = batch['num_nodes']

        # Forward pass
        scores, encoded_trees = self.model(pfp, rxnfp, adjacency_list, node_order, edge_order, num_nodes)

        output = {'scores': scores.view(-1, ).tolist()}

        if clustering:
            encoded_trees = [encoded_trees[i, :].detach().cpu().numpy() for i in range(len(trees))]
            clusters = self._cluster_encoded_trees(encoded_trees, scores=scores, cluster_method=cluster_method,
                                                   min_samples=min_samples, min_cluster_size=min_cluster_size)
            output['encoded_trees'] = encoded_trees
            output['clusters'] = clusters

        # scores, encoded_trees, clusters(optional)
        return output

    def score_record(self, record, min_samples=5, min_cluster_size=5):
        all_trees = [record['true_data']['tree']] + [r['tree'] for r in record['generated_paths']]
        output = self.scorer(all_trees, clustering=True,
                             min_samples=min_samples, min_cluster_size=min_cluster_size)

        record_output = {'true_data': {'scores': output['scores'][0],
                                       'encoded_trees': output['encoded_trees'][0],
                                       'clusters': output['clusters'][0]},
                         'generated_paths': {'scores': output['scores'][1:],
                                             'encoded_trees': output['encoded_trees'][1:],
                                             'clusters': output['clusters'][1:]}}
        return record_output

    def _cluster_encoded_trees(self, encoded_trees, scores=None, cluster_method='hdbscan',
                               min_samples=5, min_cluster_size=5):
        if not encoded_trees:
            return []
        res = []
        if 'hdbscan' == cluster_method:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        min_samples=min_samples,
                                        gen_min_span_tree=False)
            clusterer.fit(encoded_trees)
            res = clusterer.labels_
            # non-clustered inputs have id -1, make them appear as individual clusters
            max_cluster = np.amax(res)
            for i in range(len(res)):
                if res[i] == -1:
                    # print(res[i])
                    max_cluster += 1
                    res[i] = max_cluster
        elif 'kmeans' == cluster_method:  # seems to be very slow
            for cluster_size in range(len(encoded_trees)):
                kmeans = cluster.KMeans(n_clusters=cluster_size + 1).fit(encoded_trees)
                if kmeans.inertia_ < 1:
                    break
            res = kmeans.labels_
        else:
            raise Exception('Fatal error: cluster_method={} is not recognized.'.format(cluster_method))

        res = [int(i) for i in res]

        if scores is not None:
            if len(scores) != len(res):
                raise Exception(
                    'Fatal error: length of score ({}) and smiles ({}) are different.'.format(len(scores), len(res)))
            best_cluster_score = {}
            for cluster_id, score in zip(res, scores):
                best_cluster_score[cluster_id] = max(
                    best_cluster_score.get(cluster_id, -float('inf')),
                    score
                )
            new_order = list(sorted(best_cluster_score.items(), key=lambda x: -x[1]))
            order_mapping = {new_order[n][0]: n for n in range(len(new_order))}
            res = [order_mapping[n] for n in res]
        return res


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = project_path + '/trained_model/treeLSTM512-fp2048.pt'
    pathway_ranker = PathwayRanker(device, fp_size=2048, lstm_size=512)
    pathway_ranker.load_weights(model_path)

    import pickle

    with open(project_path + '/data/pathway_test_example.pkl', 'rb') as f:
        data = []
        for _ in range(5):
            data.append(pickle.load(f))


    output = pathway_ranker.scorer([d['tree'] for d in data[0]['generated_paths']], clustering=True)
