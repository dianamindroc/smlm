import argparse

import torch
import torchvision
import torch.optim as optim
import wandb
import time
import os
from shutil import copyfile
from torchvision import transforms
import numpy as np
import random
import open3d as o3d


from helpers.data import get_highest_shape
from helpers.logging import create_log_folder
from helpers.visualization import print_pc, save_plots
from model_architectures import pcn, folding_net, pointr, losses
from model_architectures.transforms import ToTensor, Padding
from model_architectures.utils import cfg_from_yaml_file, l1_cd_metric
from dataset.SMLMDataset import Dataset
from dataset.ShapeNet import ShapeNet
from dataset.SMLMSimulator import DNAOrigamiSimulator
#from model_architectures.chamfer_distances import ChamferDistanceL2, ChamferDistanceL1
from model_architectures.pointr import validate
#from chamferdist import ChamferDistance
import warnings
warnings.filterwarnings("ignore")

def train(config, ckpt=None):
    wandb.login()
    torch.backends.cudnn.benchmark = True
    config = cfg_from_yaml_file(config)
    train_config = {
        "lr": config.train.lr,
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "num_epochs": config.train.num_epochs,
        #"cd_loss": ChamferDistance(),
        "early_stop_patience": config.train.early_stop_patience,
        'momentum': config.train.momentum,
        'momentum2': config.train.momentum2,
        'scheduler_type': config.train.scheduler_type,
        'gamma': config.train.gamma,
        'step_size': config.train.step_size,
        'coarse_dim': config.pcn_config.coarse_dim,
        'fine_dim': config.pcn_config.fine_dim,
    }
    log_dir = create_log_folder(config.train.log_dir, config.model)
    print('Loading Data...')
    root_folder = config.dataset.root_folder
    classes = config.dataset.classes
    suffix = config.dataset.suffix
    #highest_shape = get_highest_shape(root_folder, classes)
    pc_transforms = transforms.Compose(
        [ToTensor()])
    #train_dataset = ShapeNet('/home/dim26fa/data/PCN', 'train', 'airplane')
    #val_dataset = ShapeNet('/home/dim26fa/data/PCN', 'valid', 'airplane')
    full_dataset = Dataset(root_folder=root_folder, suffix=suffix, transform=pc_transforms, classes_to_use=classes, data_augmentation=True, remove_part_prob = 0.3)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers'], collate_fn=custom_collate_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False, num_workers=train_config['num_workers'], collate_fn=custom_collate_fn)
    print("Dataset loaded!")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)

    # model
    model = pcn.PCN(num_dense=16384, latent_dim=1024, grid_size=4).to(device)

    # optimizer
    #optimizer = optim.Adam(model.parameters(), lr=train_config['lr'], betas=(0.9, 0.999))
    #lr_schedual = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
    optimizer = optim.Adam(model.parameters(), lr=train_config['lr'], betas=(train_config['momentum'], train_config['momentum2']), weight_decay = train_config['gamma'])

    if train_config['scheduler_type'] != 'None':
        if train_config['scheduler_type'] == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=train_config['step_size'], gamma=train_config['gamma'])
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
    print('Scheduler activated')

    if ckpt is not None:
        model.load_state_dict(torch.load(ckpt))
        print('Model loaded from ', ckpt)
        #model.first_conv.requires_grad_(False)
        #model.second_conv.requires_grad_(False)
        #model.mlp.requires_grad_(False)
        #print('Frozen all but last layer')

    wandb.init(project='autoencoder training', name='PCN_tetranup', config=train_config)
    wandb.config['optimizer'] = optimizer

    # training
    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    train_step, val_step = 0, 0
    for epoch in range(1, train_config['num_epochs'] + 1):
        # hyperparameter alpha
        if train_step < 10000:
            alpha = 0.01
        elif train_step < 20000:
            alpha = 0.1
        elif train_step < 50000:
            alpha = 0.5
        else:
            alpha = 1.0

        # training
        model.train()
        #### ShapeNet dataset
        #for i, (p, c) in enumerate(train_dataloader):
        #    p = p.permute(0, 2, 1)
        #    p, c = p.to(device), c.to(device)

        #### SMLMDataset
        for i, data in enumerate(train_dataloader):
            c = data['pc']
            p = data['partial_pc']
            for j in range(len(c)):
                c = c[j].to(device).unsqueeze(0)  # Add batch dimension
                p = p[j].to(device).unsqueeze(0)  # Add batch dimension
                p = p.permute(0, 2, 1)  # Adjust dimensions if needed

                optimizer.zero_grad()

                # Forward propagation
                coarse_pred, dense_pred = model(p)

                # Calculate loss
                loss1 = losses.cd_loss_l1(coarse_pred, c)
                loss2 = losses.cd_loss_l1(dense_pred, c)
                loss = loss1 + alpha * loss2

                # Backward propagation
                loss.backward()
                optimizer.step()

                train_step += 1  # Assuming train_step is a counter for logging or similar
        print("Train Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, train_config['num_epochs'], loss * 1e3))
        wandb.log({'train_l1_cd': loss * 1e3}, step=train_step)
        export_ply(os.path.join(log_dir, '{:03d}_train_pred.ply'.format(epoch)), dense_pred[0].detach().cpu().numpy())
        #export_ply(os.path.join(log_dir, '{:03d}.ply'.format(i)), np.transpose(dense_pred[0].detach().cpu().numpy(), (1,0)))
        #lr_schedual.step()

        # evaluation
        model.eval()
        total_cd_l1 = 0.0
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization
            #### ShapeNet dataset
            #for i, (p, c) in enumerate(val_dataloader):
            #    p = p.permute(0, 2, 1)
            #    p, c = p.to(device), c.to(device)

            #### SMLMDataset
            for i, data in enumerate(val_dataloader):
                c = data['pc']
                p = data['partial_pc']
                for j in range(len(c)):
                    c = c[j].to(device).unsqueeze(0)  # Add batch dimension
                    p = p[j].to(device).unsqueeze(0)  # Add batch dimension
                    p = p.permute(0, 2, 1)  # Adjust dimensions if needed

                    coarse_pred, dense_pred = model(p)
                    total_cd_l1 += losses.cd_loss_l1(dense_pred, c).item()

                    if i == rand_iter and j == 0:  # Adjust visualization code as needed
                        input_pc = p[0].detach().cpu().numpy()
                        input_pc = np.transpose(input_pc, (1, 0))
                        output_pc = dense_pred[0].detach().cpu().numpy()
                        export_ply(os.path.join(log_dir, '{:03d}_input.ply'.format(epoch)), input_pc)
                        export_ply(os.path.join(log_dir, '{:03d}_output.ply'.format(epoch)), output_pc)
            total_cd_l1 /= len(val_dataset)
            scheduler.step(total_cd_l1)
            val_step += 1
            wandb.log({'val_l1_cd': total_cd_l1 * 1e3}, step=val_step)

            print("Validate Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, train_config['num_epochs'], total_cd_l1 * 1e3))

        if total_cd_l1 < best_cd_l1:
            best_epoch_l1 = epoch
            best_cd_l1 = total_cd_l1
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_l1_cd.pth'))


    print('Best l1 cd model in epoch {}, the minimum l1 cd is {}'.format(best_epoch_l1, best_cd_l1 * 1e3))

def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)

def custom_collate_fn(batch):
    # Initialize containers for batched data
    pcs = []
    partial_pcs = []
    labels = []
    paths = []

    # Iterate through each sample in the batch
    for item in batch:
        pcs.append(item['pc'])
        partial_pcs.append(item['partial_pc'])
        labels.append(item['label'])
        paths.append(item['path'])

    # Handle other fields (labels, paths) as required
    labels = torch.tensor(labels)
    # Paths don't need to be tensor

    # No need to pad since we are dealing with a custom batch handling
    batched_data = {'pc': pcs, 'partial_pc': partial_pcs, 'label': labels, 'path': paths}

    return batched_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PCN')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--log_dir', type=str, default='log', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--epochs', type=int, default=200, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader')
    parser.add_argument('--coarse_loss', type=str, default='cd', help='loss function for coarse point cloud')
    parser.add_argument('--num_workers', type=int, default=6, help='num_workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--log_frequency', type=int, default=10, help='Logger frequency in every epoch')
    parser.add_argument('--save_frequency', type=int, default=10, help='Model saving frequency')
    params = parser.parse_args()

    train(params)