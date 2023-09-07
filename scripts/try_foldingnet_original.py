import argparse
import os
import time

import torch
import torch.optim as optim

from dataset.SMLMDataset import Dataset
from model_architectures import folding_net
from chamfer_distance.chamfer_distance import ChamferDistance
from model_architectures.transforms import ToTensor, Padding
from torchvision import transforms
from helpers.visualization import print_pc, save_plots
from helpers.data import get_highest_shape

root_folder = '/home/dim26fa/data/shapenet'
suffix = '.pts'
classes = 'all'
lr = 0.0001
weight_decay = 1e-6
sample_to_save_path = None
highest_shape = get_highest_shape(root_folder, classes)
pc_transforms = transforms.Compose([Padding(highest_shape), ToTensor()])
full_dataset = Dataset(root_folder=root_folder, suffix=suffix, classes_to_use=classes, data_augmentation=False, transform=pc_transforms)
train_config = {
                'batch_size': 1,
                'num_workers': 4
                }
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True,
                                               num_workers=train_config['num_workers'])
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False,
                                             num_workers=train_config['num_workers'])
# device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model
autoendocer = folding_net.AutoEncoder()
autoendocer.to(device)

# loss function
cd_loss = ChamferDistance()
# optimizer
optimizer = optim.Adam(autoendocer.parameters(), lr=lr, betas=[0.9, 0.999], weight_decay=weight_decay)

batches = int(len(train_dataset) / train_config['batch_size'] + 0.5)

min_cd_loss = 1e3
best_epoch = -1
epochs = 287
log_dir = '/home/dim26fa/coding/training/foldingnet_test_original/batch1'

print('\033[31mBegin Training...\033[0m')
for epoch in range(1, epochs + 1):
    # training
    start = time.time()
    autoendocer.train()
    for i, data in enumerate(train_dataloader):
        point_clouds= data['pc']
        point_clouds = point_clouds.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        recons = autoendocer(point_clouds)
        dist1, dist2 = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
        ls = torch.max(torch.cat([dist1.mean(dim=-1, keepdim=True), dist2.mean(dim=-1, keepdim=True)], dim=1), dim=1)[0].sum()

        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch {}/{} with iteration {}/{}: CD loss is {}.'.format(epoch, epochs, i + 1, batches, ls.item()))

    # evaluation
    autoendocer.eval()
    total_cd_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            if sample_to_save_path is None:
                    sample_to_save_path = data['path'][0]
            point_clouds = data['pc']
            point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds = point_clouds.to(device)
            recons = autoendocer(point_clouds)
            dist1, dist2 = cd_loss(point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            ls = torch.max(torch.cat([dist1.mean(dim=-1, keepdim=True), dist2.mean(dim=-1, keepdim=True)], dim=1), dim=1)[0].sum()

            total_cd_loss += ls.item()
            if sample_to_save_path in data['path']:
                    index = data['path'].index(sample_to_save_path)
                    sample_to_save = [data['pc'][index], recons.permute(0, 2, 1)[index]]
    # calculate the mean cd loss
    mean_cd_loss = total_cd_loss / len(test_dataset)

    # records the best model and epoch
    if mean_cd_loss < min_cd_loss:
        min_cd_loss = mean_cd_loss
        best_epoch = epoch
        torch.save(autoendocer.state_dict(), os.path.join(log_dir, 'model_lowest_cd_loss.pth'))
        save_plots(epoch, mean_cd_loss, sample_to_save, log_dir)

    # save the model every 100 epochs
    if (epoch) % 100 == 0:
        torch.save(autoendocer.state_dict(), os.path.join(log_dir, 'model_epoch_{}.pth'.format(epoch)))

    end = time.time()
    cost = end - start

    print('\033[32mEpoch {}/{}: reconstructed Chamfer Distance is {}. Minimum cd loss is {} in epoch {}.\033[0m'.format(
        epoch, epochs, mean_cd_loss, min_cd_loss, best_epoch))
    print('\033[31mCost {} minutes and {} seconds\033[0m'.format(int(cost // 60), int(cost % 60)))