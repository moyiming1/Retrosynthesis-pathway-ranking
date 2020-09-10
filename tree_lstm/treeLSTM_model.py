import os, sys

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

import torch
import numpy
import torch.nn.functional as F

from .model import PathwayRankingModel


def loss_softmax(score, num_trees, device=torch.device('cpu')):
    """softmax ranking loss"""
    score_split = torch.split(score, num_trees, dim=0)
    loss_split = [F.cross_entropy(s.view(1, s.shape[0]),
                                  torch.tensor([0], dtype=torch.int64, device=device))
                  for s in score_split]
    loss = torch.mean(torch.stack(loss_split))
    return loss

def topk_accuracy(score, num_trees, topk=(1, 3, 5)):
    """Computes the precision@k for the specified values of k"""
    score_split = torch.split(score, num_trees, dim=0)
    scores = [numpy.asarray(s.tolist()) for s in score_split]

    accuracy = []
    for sc in scores:
        # print(sc.reshape(-1,).shape)
        index = numpy.argsort(-1 * sc.reshape(-1, ))
        accuracy_k = []
        for top in topk:
            if 0 in list(index)[0:top]:
                accuracy_k.append(1)
            else:
                accuracy_k.append(0)
        accuracy.append(accuracy_k)

    res = []
    for i, k in enumerate(topk):
        temp = [line[i] for line in accuracy]
        res.append(sum(temp) / len(temp))
    # print(res)
    return res


if __name__ == '__main__':
    import pickle
    with open(project_path + '/data/pathway_train_example.pkl', 'rb') as f:
        data = []
        for _ in range(20):
            data.append(pickle.load(f))

    fp_size = 2048
    lstm_size = 512

    from features.tree_to_treeLSTM_input import convert_multiple_records_into_batch

    batch = convert_multiple_records_into_batch(data[0:4], to_tensor=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pfp = batch['pfp'].to(device)
    rxnfp = batch['rxnfp'].to(device)
    adjacency_list = batch['adjacency_list'].to(device)
    node_order = batch['node_order'].to(device)
    edge_order = batch['edge_order'].to(device)
    num_trees = batch['num_trees']

    model = PathwayRankingModel(fp_size, lstm_size).to(device)
    score = model(pfp, rxnfp,
                  adjacency_list, node_order,
                  edge_order, batch['num_nodes'])

    loss = loss_softmax(score, batch['num_trees'], device=device)

    print(topk_accuracy(score, num_trees))
