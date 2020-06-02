import os, sys

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

import torch
import math

from tree_lstm.treeLSTM_model import topk_accuracy
from utils.scheduler import NoamLR, SinexpLR
import time


def train(model,
          loss_fn,
          optimizer,
          data_loader,
          scheduler,
          num_data,
          batch_size,
          device,
          n_iter,
          epoch,
          log_frequency):
    t0 = time.time()
    total_step = math.ceil(num_data // batch_size)
    loss_sum = 0
    metric_sum = [0] * 5

    step_count = 0
    for i, batch in enumerate(data_loader):
        pfp = batch['pfp'].to(device)
        rxnfp = batch['rxnfp'].to(device)
        adjacency_list = batch['adjacency_list'].to(device)
        node_order = batch['node_order'].to(device)
        edge_order = batch['edge_order'].to(device)
        num_nodes = batch['num_nodes']
        num_trees = batch['num_trees']
        batch_size = batch['batch_size']
        # print('converting to device: ', time.time()-t00, ' s')
        # Forward pass
        scores = model(pfp, rxnfp, adjacency_list, node_order, edge_order, num_nodes)
        loss = loss_fn(scores, num_trees, device=device)

        acc = topk_accuracy(scores, num_trees, topk=(1, 5, 10, 30, 100))
        # print('model forward: ', time.time() - t00, ' s')
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('optimizing: ', time.time() - t00, ' s')
        if isinstance(scheduler, NoamLR) or isinstance(scheduler, SinexpLR):
            scheduler.step()

        # calculate status
        loss_sum += loss.item()
        metric_sum = [x + y for x, y in zip(metric_sum, acc)]
        step_count += 1

        n_iter += batch_size

        # print log
        if (i + 1) % log_frequency == 0:
            loss_avg = loss_sum / step_count
            metric_mean = [x / step_count for x in metric_sum]
            lr = scheduler.get_lr()[0]

            loss_sum = 0
            metric_sum = [0] * 5
            step_count = 0

            t = time.time()
            # print(scores)
            print('Training: Epoch {}, step {}/{}, time {:.2f} h/{:.2f} h, '
                  'Loss: {:.4f}. Acc: top1 {:.3f}, top5 {:.3f}, '
                  'top10 {:.3f}, top30 {:.3f}, top100 {:.3f}, lr {:.5f}'
                  .format(epoch, i, total_step,
                          (t - t0) / 3600, (t - t0) * total_step / (i + 1) / 3600,
                          loss_avg, *metric_mean, lr))
    return n_iter


def valid(model,
          loss_fn,
          data_loader,
          num_data,
          batch_size,
          device,
          epoch,
          log_frequency):
    loss_sum = 0
    metric_sum = [0] * 5

    loss_avg = 0
    metric_mean = [0] * 5
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            pfp = batch['pfp'].to(device)
            rxnfp = batch['rxnfp'].to(device)
            adjacency_list = batch['adjacency_list'].to(device)
            node_order = batch['node_order'].to(device)
            edge_order = batch['edge_order'].to(device)
            num_nodes = batch['num_nodes']
            num_trees = batch['num_trees']

            # Forward pass
            scores = model(pfp, rxnfp, adjacency_list, node_order, edge_order, num_nodes)
            loss = loss_fn(scores, num_trees, device=device)
            acc = topk_accuracy(scores, num_trees, topk=(1, 5, 10, 30, 100))

            # calculate status
            loss_sum += loss.item()
            metric_sum = [x + y for x, y in zip(metric_sum, acc)]
            # print log
            if (i + 1) % log_frequency == 0:
                loss_avg = loss_sum / (i + 1)
                metric_mean = [x / (i + 1) for x in metric_sum]

                print('Validation: Epoch {}, step {}/{}, Loss: {:.4f}. Acc: top1 {:.3f}, top5 {:.3f}, '
                      'top10 {:.3f}, top30 {:.3f}, top100 {:.3f}'
                      .format(epoch, i, math.ceil(num_data // batch_size), loss_avg, *metric_mean))
    return loss_avg, metric_mean


def test(model,
         loss_fn,
         data_loader,
         batch_size,
         device,
         log_frequency,
         num_data):
    model.eval()
    topk = (1, 5, 10, 30, 50, 100)
    loss_sum = 0
    metric_sum = [0] * len(topk)

    loss_avg = 0
    metric_mean = [0] * len(topk)
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            pfp = batch['pfp'].to(device)
            rxnfp = batch['rxnfp'].to(device)
            adjacency_list = batch['adjacency_list'].to(device)
            node_order = batch['node_order'].to(device)
            edge_order = batch['edge_order'].to(device)
            num_nodes = batch['num_nodes']
            num_trees = batch['num_trees']
            # Forward pass
            scores = model(pfp, rxnfp, adjacency_list, node_order, edge_order, num_nodes)
            loss = loss_fn(scores, num_trees, device=device)
            acc = topk_accuracy(scores, num_trees, topk=topk)

            # calculate status
            loss_sum += loss.item()
            metric_sum = [x + y for x, y in zip(metric_sum, acc)]
            # print log
            if (i + 1) % log_frequency == 0:
                loss_avg = loss_sum / (i + 1)
                metric_mean = [x / (i + 1) for x in metric_sum]

                print('Test: step {}/{}, Loss: {:.4f}. Acc: top1 {:.3f}, top5 {:.3f}, '
                      'top10 {:.3f}, top30 {:.3f}, top50 {:.3f}, top100 {:.3f}'
                      .format(i, math.ceil(num_data // batch_size), loss_avg, *metric_mean))
    return loss_avg, metric_mean
