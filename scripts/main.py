from model_architectures import pcn, folding_net, pointr
from model_architectures.utils import cfg_from_yaml_file
from dataset.SMLMDataset import Dataset
from model_architectures.chamfer_distances import ChamferDistanceL2

import torch
import torch.optim as optim
import wandb
import time
import os

wandb.login()
wandb.init(project='autoencoder training', name='second run')

config = cfg_from_yaml_file('configs/config.yaml')

# Load models
model_pcn = pcn.PCN(16384, 1024)
model_fold = folding_net.AutoEncoder()
model_pointr = pointr.AdaPoinTr(config.model)

################ Try Foldingnet ######################

# Set training params
lr = 0.0001
batch_size = 32
num_workers = 2
num_epochs = 100
log_dir = '/mnt/c/Users/diana/PhD/test_dna_smlm/training'
optimizer = optim.Adam(model_fold.parameters(), lr=lr, betas=[0.9, 0.999])
cd_loss = ChamferDistanceL2()

# Load params in wandb config
wandb.config.learning_rate = lr
wandb.config.batch_size = batch_size
wandb.config.num_workers = num_workers
wandb.config.num_epochs = num_epochs
wandb.config.optimizer = optimizer
wandb.config.loss = cd_loss

# Load data
full_dataset = Dataset(root_folder='/mnt/c/Users/diana/PhD/data/cube', filename='smlm_sample',suffix='.csv')
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Setup device and move model to it
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_fold = model_fold.to(device)

batches = int(len(train_dataset) / batch_size + 0.5)
best_epoch = -1
min_cd_loss = 1e3

print('\033[31mBegin Training...\033[0m')
for epoch in range(1, num_epochs + 1):
    # training
    start = time.time()
    model_fold.train()
    total_cd_loss_train = 0
    for i, data in enumerate(train_dataloader):
        point_clouds = data
        point_clouds = point_clouds.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        recons = model_fold(point_clouds)
        ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        wandb.log({"training_step_loss": ls})
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()
        total_cd_loss_train += ls

        if (i + 1) % 100 == 0:
            print('Epoch {}/{} with iteration {}/{}: CD loss is {}.'.format(epoch, num_epochs, i + 1, batches,
                                                                            ls[0] / len(point_clouds)))
    mean_cd_loss_train = total_cd_loss_train / len(test_dataset)
    wandb.log({"training_loss": mean_cd_loss_train})

    # evaluation
    model_fold.eval()
    total_cd_loss_test = 0
    with torch.no_grad():
        for data in test_dataloader:
            point_clouds = data
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds = point_clouds.to(device)
            recons = model_fold(point_clouds)
            ls = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            total_cd_loss_test += ls
            wandb.log({"test_step_loss": ls})

    # calculate the mean cd loss
    mean_cd_loss = total_cd_loss_test / len(test_dataset)
    wandb.log({"test_loss": mean_cd_loss})

    # records the best model and epoch
    if mean_cd_loss < min_cd_loss:
        min_cd_loss = mean_cd_loss
        best_epoch = epoch
        torch.save(model_fold.state_dict(), os.path.join(log_dir, 'model_lowest_cd_loss.pth'))

    # save the model every 100 epochs
    if epoch % 100 == 0:
        torch.save(model_fold.state_dict(), os.path.join(log_dir, 'model_epoch_{}.pth'.format(epoch)))

    end = time.time()
    cost = end - start

    print('\033[32mEpoch {}/{}: reconstructed Chamfer Distance is {}. Minimum cd loss is {} in epoch {}.\033[0m'.format(
        epoch, num_epochs, mean_cd_loss, min_cd_loss, best_epoch))
    print('\033[31mCost {} minutes and {} seconds\033[0m'.format(int(cost // 60), int(cost % 60)))