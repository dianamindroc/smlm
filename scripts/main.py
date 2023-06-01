from helpers.data import get_highest_shape
from model_architectures import pcn, folding_net, pointr
from model_architectures.utils import cfg_from_yaml_file
from dataset.SMLMDataset import Dataset
from model_architectures.chamfer_distances import ChamferDistanceL2

import torch
import torch.optim as optim
import wandb
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

wandb.login()


#config = cfg_from_yaml_file('configs/config.yaml')

# Load models
model_pcn = pcn.PCN(16384, 1024)
model_fold = folding_net.AutoEncoder()
#model_pointr = pointr.AdaPoinTr(config.model)

root_folder = '/mnt/c/Users/diana/PhD/data/datasets'
classes = ['cube', 'pyramid']
#highest_shape = get_highest_shape(root_folder, classes)
highest_shape = 250
################ Try Foldingnet ######################

# Set training params
train_config = {
    "lr": 0.0001,
    "batch_size": 32,
    "num_workers": 4,
    "num_epochs": 200,
    "cd_loss": ChamferDistanceL2(),
    "early_stop_patience": 5
}

optimizer = optim.Adam(model_fold.parameters(), lr=train_config['lr'], betas=[0.9, 0.999])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
log_dir = '/mnt/c/Users/diana/PhD/test_dna_smlm/training'

# Load params in wandb config
wandb.init(project='autoencoder training', name='fifth run', config=train_config)
wandb.config['optimizer'] = optimizer
wandb.config['scheduler'] = scheduler



# Load data
full_dataset = Dataset(root_folder=root_folder, filename='smlm_sample', suffix='.csv', classes=classes, highest_shape=highest_shape)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'])
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=train_config['num_workers'])

# Setup device and move model to it
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_fold = model_fold.to(device)

batches = int(len(train_dataset) / train_config['batch_size'] + 0.5)
best_epoch = -1
min_cd_loss = 1e3
no_improvement = 0
current_lr = train_config['lr']

print('\033[31mBegin Training...\033[0m')
for epoch in range(1, train_config['num_epochs'] + 1):
    # training
    start = time.time()
    model_fold.train()
    total_cd_loss_train = 0
    for i, data in enumerate(train_dataloader):
        point_clouds = data['pc']
        point_clouds = point_clouds.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        recons = model_fold(point_clouds)
        ls = train_config['cd_loss'](point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        wandb.log({"training_step_loss": ls})
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        total_cd_loss_train += ls

        if (i + 1) % 100 == 0:
            print('Epoch {}/{} with iteration {}/{}: CD loss is {}.'.format(epoch, train_config['num_epochs'], i + 1,
                                                                            batches, ls[0] / len(point_clouds)))
    mean_cd_loss_train = total_cd_loss_train / len(train_dataset)
    wandb.log({"training_loss": mean_cd_loss_train})

    # evaluation
    model_fold.eval()
    total_cd_loss_val = 0
    with torch.no_grad():
        for data in val_dataloader:
            point_clouds = data['pc']
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds = point_clouds.to(device)
            recons = model_fold(point_clouds)
            ls = train_config['cd_loss'](point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            total_cd_loss_val += ls
            wandb.log({"val_step_loss": ls})

    # calculate the mean cd loss
    mean_cd_loss = total_cd_loss_val / len(val_dataset)
    scheduler.step(mean_cd_loss)
    if current_lr is not optimizer.param_groups[0]['lr'] and train_config['early_stop_patience'] - no_improvement <= 2:
        # we want to let the training run a few epochs with the new learning rate, therefore we add this if-clause
        no_improvement -= 2
        current_lr = optimizer.param_groups[0]['lr']
    wandb.log({"val_loss": mean_cd_loss})

    # records the best model and epoch
    if mean_cd_loss < min_cd_loss:
        no_improvement = 0
        min_cd_loss = mean_cd_loss
        best_epoch = epoch
        torch.save(model_fold.state_dict(), os.path.join(log_dir, 'model_lowest_cd_loss.pth'))
    else:
        no_improvement += 1
        if no_improvement > train_config['early_stop_patience']:
            print('\033[32mEarly stoppage at epoch {}/{}: reconstructed Chamfer Distance is {}. Minimum cd loss is {} in epoch {}.\033[0m'.format(
                    epoch, train_config['num_epochs'], mean_cd_loss, min_cd_loss, best_epoch))
            wandb.log({"epoch_stoppage": epoch})
            wandb.save(os.path.join(wandb.run.dir, "model.h5"))
            wandb.finish()
            break

    # save the model every 100 epochs
    if epoch % 100 == 0:
        torch.save(model_fold.state_dict(), os.path.join(log_dir, 'model_epoch_{}.pth'.format(epoch)))

    end = time.time()
    cost = end - start

    print('\033[32mEpoch {}/{}: reconstructed Chamfer Distance is {}. Minimum cd loss is {} in epoch {}.\033[0m'.format(
        epoch, train_config['num_epochs'], mean_cd_loss, min_cd_loss, best_epoch))
    print('\033[31mCost {} minutes and {} seconds\033[0m'.format(int(cost // 60), int(cost % 60)))
wandb.save(os.path.join(wandb.run.dir, "model.h5"))
wandb.finish()
