import torch
import numpy
import torch.nn.functional as F

def loss_marginranking(score, num_trees, device=torch.device('cpu')):

    score_split = torch.split(score, num_trees)

    loss_split = [F.margin_ranking_loss(s[0] * torch.ones(s.shape[0] - 1, device=device),
                                        s[1:],
                                        torch.ones(s.shape[0] - 1, device=device),
                                        margin=0.05,
                                        reduction='mean') for s in score_split]

    loss = torch.mean(torch.stack(loss_split))
    return loss


def loss_softmax(score, num_trees, device=torch.device('cpu')):
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


class DSNN(torch.nn.Module):
    '''
    SC score baseline model
    '''

    def __init__(self, in_features, out_features, hidden_size=256, layers=5, dropout=0.0, linear_pool='mean'):
        super(DSNN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.linear_pool = linear_pool

        dropout = torch.nn.Dropout(dropout)
        activation = torch.nn.ReLU()

        modules = []
        for i in range(layers):
            if i == 0:
                modules.extend([
                    dropout,
                    torch.nn.Linear(self.in_features, hidden_size),
                    activation
                ])
            if i < layers -1 :
                modules.extend([
                    dropout,
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    activation
                ])
                hidden_size = hidden_size // 2
            else:
                modules.extend([
                    dropout,
                    torch.nn.Linear(hidden_size, self.out_features),
                ])

        self.nn = torch.nn.Sequential(*modules)

    def forward(self, features):
        scores = self.nn(features)

        return scores


























































































































































































