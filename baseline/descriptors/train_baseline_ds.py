import os, sys

project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_path)

import torch

from baseline.descriptors import train, DSNN, loss_marginranking, valid, loss_softmax

from utils.save_load import save_checkpoint
from utils.scheduler import build_lr_scheduler
from features import get_ds_dataloader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import multiprocessing

print('cpu count:', multiprocessing.cpu_count())
print('cuda available:', torch.cuda.is_available())

# configurations
fp_loader = True
sigmoid = True
loss = 'softmax'
linear_pool = 'min'
label = 'ds_baseline_256_5'
layers = 5
# Hyper-parameters

num_workers = 10
num_epochs = 10
batch_size = 32
log_freq = 20
# paths
# data_path = '/data/yanfei/Projects/Pathway_ranking/data/all_training_data/tree_lstm'
# data_path = '/mnt/data/home/yiming/Projects/data/pathway_ranking/tree_lstm_fp_input'
data_path = '/home/yanfeig/Projects/pathway/data/all_training_data/all_training_data_with_sc'
#data_path = '/Users/yanfei/Projects/path_ranking/pathway_ranking/data/debug'
# data_path = 'data/'
# data_path = '/nobackup/users/yimingmo/project/data/pathway_ranking/pathway_data'
# data_path = '/mnt/data/home/yiming/Projects/data/pathway_ranking/curated_data'
# data_path = '/mnt/data/home/yiming/Projects/pathway_ranking/model/tree_lstm/data'
# save_path = '/nobackup/users/yimingmo/project/data/pathway_ranking/training_results'
#save_path = '/Users/yanfei/Projects/path_ranking/pathway_ranking/useless'
#save_path = '/data/yanfei/Projects/Pathway_ranking/model/results/tree_lstm_1'
save_path = '/home/yanfeig/Projects/pathway/model/results/baseline'

if loss == 'softmax':
    loss_fn = loss_softmax
elif loss == 'margin':
    loss_fn = loss_marginranking
elif loss == 'binary':
    loss_fn = loss_binary
else:
    raise ValueError('incorrect loss fn selection')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scheduler_name = 'Sinexp'
scheduler_args = {'init_lr': 0.01, 'final_lr': 0.001}

# Data loader
# add dataloader
valid_loader = get_ds_dataloader(data_path=data_path, job_type='valid',
                                 batch_size=batch_size, num_workers=num_workers, device=device)
test_loader = get_ds_dataloader(data_path=data_path, job_type='test',
                                batch_size=batch_size, num_workers=num_workers, device=device)

# data information
train_data_size = 207150
valid_data_size = 25530

# build model
model = DSNN(9, 1, layers=layers).to(device)

# Loss and optimizer
optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': scheduler_args['init_lr']}])

scheduler = build_lr_scheduler(optimizer, scheduler_name, scheduler_args,
                               train_data_size, batch_size, num_epochs)

writer = SummaryWriter(os.path.join(save_path, 'tensorboard/train_03292020_' + label))

# args for saving the model
model_args = {
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
    'loss': loss,
    'sigmoid_output': sigmoid,
    'fp_loader': fp_loader
}
print(model_args)
# Train the model
n_iter = 0
best_valid_loss = float('inf')
for epoch in range(num_epochs):
    train_loader = get_ds_dataloader(data_path=data_path, job_type='train',
                                     batch_size=batch_size, num_workers=num_workers, device=device)

    model.train()

    n_iter = train(model, loss_fn, optimizer, train_loader, scheduler,
                   num_data=train_data_size, batch_size=batch_size,
                   device=device, n_iter=n_iter, writer=writer,
                   log_frequency=log_freq, epoch=epoch)

    model.eval()
    valid_loss, valid_metric = valid(model, loss_fn, valid_loader, num_data=valid_data_size,
                                     batch_size=batch_size, device=device, epoch=epoch,
                                     log_frequency=log_freq)
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step()

    writer.add_scalar('Loss/valid', valid_loss, n_iter)
    writer.add_scalar('Acc/top1/valid', valid_metric[0], n_iter)
    writer.add_scalar('Acc/top5/valid', valid_metric[1], n_iter)
    writer.add_scalar('Acc/top10/valid', valid_metric[2], n_iter)
    writer.add_scalar('Acc/top30/valid', valid_metric[3], n_iter)
    writer.add_scalar('Acc/top100/valid', valid_metric[4], n_iter)

    save_checkpoint(os.path.join(save_path, 'trained_model_' + label + '_epoch_{}.pt'.format(epoch)), model, model_args)

    if valid_loss < best_valid_loss:
        save_checkpoint(os.path.join(save_path, 'best_model_' + label + '.pt'), model, model_args)
        best_valid_loss = valid_loss

writer.close()

