import os, sys

project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_path)

import torch
from tree_lstm.treeLSTM_model import PathwayRankingModel, loss_softmax
from tree_lstm.train_valid_test import train, valid
from utils.save_load import save_checkpoint
from utils.scheduler import build_lr_scheduler
from tree_lstm.tree_generator import get_tree_dataloader as dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing

print('cpu count:', multiprocessing.cpu_count())
print('cuda available:', torch.cuda.is_available())

# configurations
label = 'treeLSTM512-fp2048'
scheduler_name = 'Sinexp'
lr_mul = 1
scheduler_args = {'init_lr': 0.002*lr_mul, 'final_lr': 0.0005*lr_mul}
loss_fn = loss_softmax
# Hyper-parameters
fp_size = 2048
lstm_size = 512
num_workers = 1  # use >1 when there are more than 1 data files for train, valid, test
num_epochs = 5
batch_size = 8  # choose base on GPU memory
log_freq = 1

# paths
data_path = project_path + '/data'
save_path = project_path + '/trained_model'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loader
valid_loader = dataloader(data_path=data_path, job_type='pathway_valid',
                          fp_size=fp_size, batch_size=batch_size, num_workers=num_workers, device=device)
test_loader = dataloader(data_path=data_path, job_type='pathway_test',
                         fp_size=fp_size, batch_size=batch_size, num_workers=num_workers, device=device)

# data information
train_data_size = 200
valid_data_size = 40

# build model
model = PathwayRankingModel(fp_size, lstm_size).to(device)

# Loss and optimizer
optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': scheduler_args['init_lr']}])

scheduler = build_lr_scheduler(optimizer, scheduler_name, scheduler_args,
                               train_data_size, batch_size, num_epochs)

# args for saving the model
model_args = {
    'fp_size': fp_size,
    'lstm_size': lstm_size,
    'num_workers': num_workers,
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'scheduler_name': scheduler_name,
    'scheduler_args': scheduler_args,
    'log_freq': log_freq,
    'train_data_size': train_data_size,
    'valid_data_size': valid_data_size,
    'data_path': data_path,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'label': label,
}
print(model_args)
# Train the model
n_iter = 0
best_valid_loss = float('inf')

for epoch in range(num_epochs):
    train_loader = dataloader(data_path=data_path, job_type='pathway_train',
                              fp_size=fp_size, batch_size=batch_size,
                              num_workers=num_workers, device=device, shuffle=True)

    model.train()

    n_iter = train(model, loss_fn, optimizer, train_loader, scheduler,
                   num_data=train_data_size, batch_size=batch_size,
                   device=device, n_iter=n_iter,
                   log_frequency=log_freq, epoch=epoch)
    model.eval()
    valid_loss, valid_metric = valid(model, loss_fn, valid_loader, num_data=valid_data_size,
                                     batch_size=batch_size, device=device, epoch=epoch,
                                     log_frequency=log_freq)
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step()

    save_checkpoint(os.path.join(save_path, 'trained_model_' + label + '_epoch_{}.pt'.format(epoch)), model, model_args)

    if valid_loss < best_valid_loss:
        save_checkpoint(os.path.join(save_path, 'best_model_' + label + '.pt'), model, model_args)
        best_valid_loss = valid_loss

