import os, sys

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

import torch
from tree_lstm.treeLSTM_model import PathwayRankingModel, loss_softmax
from tree_lstm.train_valid_test import test
from tree_lstm.tree_generator import get_tree_dataloader as dataloader
import multiprocessing

print('cpu count:', multiprocessing.cpu_count())
print('cuda available:', torch.cuda.is_available())

# paths
data_path = project_path + '/data'
save_path = project_path + '/trained_model'
load_model_path = project_path + '/trained_model/treeLSTM512-fp2048.pt'

saved_model = torch.load(load_model_path, map_location=lambda storage, loc: storage)
args, loaded_state_dict = saved_model['args'], saved_model['state_dict']

# configurations
# Hyper-parameters
fp_size = args['fp_size']
lstm_size = args['lstm_size']
num_workers = 1
batch_size = 8
log_freq = 1
loss_fn = loss_softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loader
test_loader = dataloader(data_path=data_path, job_type='pathway_test',
                         fp_size=fp_size, batch_size=batch_size, num_workers=num_workers, device=device)

# data information
test_data_size = 40

# build model
model = PathwayRankingModel(fp_size, lstm_size).to(device)
model.load_state_dict(loaded_state_dict)
model.eval()
# test the model
test_loss, test_metric = test(model, loss_fn, test_loader, num_data=test_data_size,
                              batch_size=batch_size, device=device,
                              log_frequency=log_freq)

print('Loss:', test_loss)
print('Top-k accuracy', test_metric)
