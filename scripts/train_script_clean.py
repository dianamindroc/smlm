import argparse
import os
import random
import warnings

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import open3d as o3d
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
from sklearn.manifold import TSNE
from torchvision import transforms

from dataset.SMLMDataset import Dataset
from dataset.ShapeNet import ShapeNet
from dataset.SMLMSimulator import DNAOrigamiSimulator
from helpers.data import get_highest_shape
from helpers.logging import create_log_folder
from model_architectures import pcn, losses
from model_architectures.transforms import ToTensor, Padding
from model_architectures.utils import cfg_from_yaml_file, set_seed

matplotlib.use('Agg')
warnings.filterwarnings("ignore")


def train(conf, exp_name=None, fixed_alpha=None):
    set_seed(42)
    wandb.login()
    torch.backends.cudnn.benchmark = True
    config = cfg_from_yaml_file(conf)
    train_config = get_train_config(config)

    log_dir = create_log_folder(config.train.log_dir, config.model)
    train_dataloader, val_dataloader = get_data_loaders(config, train_config)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)

    model = get_model(config, train_config, device)
    optimizer_recon, optimizer_mlp = get_optimizers(model, train_config)
    scheduler_recon, scheduler_class = get_schedulers(optimizer_recon, optimizer_mlp, train_config)

    if os.path.exists(train_config['ckpt']):
        model.load_state_dict(torch.load(train_config['ckpt']))
        print('Model loaded from ', train_config['ckpt'])
        freeze_model_layers(model)

    setup_wandb(exp_name, train_config, optimizer_recon, optimizer_mlp, fixed_alpha)

    train_and_evaluate(model, train_dataloader, val_dataloader, train_config, log_dir, device, scheduler_recon,
                       scheduler_class)


def get_train_config(config):
    return {
        "lr": config.train.lr,
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "num_epochs": config.train.num_epochs,
        "early_stop_patience": config.train.early_stop_patience,
        'momentum': config.train.momentum,
        'momentum2': config.train.momentum2,
        'scheduler_type': config.train.scheduler_type,
        'scheduler_factor': config.train.scheduler_factor,
        'scheduler_patience': config.train.scheduler_patience,
        'gamma': config.train.gamma,
        'step_size': config.train.step_size,
        'coarse_dim': config.pcn_config.coarse_dim,
        'fine_dim': config.pcn_config.fine_dim,
        'remove_part_prob': config.dataset.remove_part_prob,
        'dataset': config.dataset.dataset_type,
        'which_dataset': config.dataset.root_folder,
        'remove_corners': config.dataset.remove_corners,
        'anisotropy': config.dataset.anisotropy,
        'anisotropy_factor': config.dataset.anisotropy_factor,
        'anisotropy_axis': config.dataset.anisotropy_axis,
        'classifier': config.train.classifier,
        'ckpt': config.train.ckpt,
        'autoencoder': config.train.autoencoder
    }


def get_data_loaders(config, train_config):
    print('Loading Data...')
    root_folder = config.dataset.root_folder
    classes = config.dataset.classes
    highest_shape = get_highest_shape(root_folder, classes)
    pc_transforms = transforms.Compose([Padding(highest_shape), ToTensor()])
    suffix = config.dataset.suffix
    full_dataset = load_dataset(train_config, root_folder, classes, pc_transforms, suffix)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True,
                                                   num_workers=train_config['num_workers'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False,
                                                 num_workers=train_config['num_workers'])
    print("Dataset loaded!")
    return train_dataloader, val_dataloader


def load_dataset(train_config, root_folder, classes, pc_transforms, suffix):
    if train_config['dataset'] == 'shapenet':
        return ShapeNet(root_folder, 'train', classes), ShapeNet(root_folder, 'valid', classes)
    elif train_config['dataset'] == 'simulate':
        dye_properties_dict = {
            'Alexa_Fluor_647': {'density_range': (10, 50), 'blinking_times_range': (10, 50),
                                'intensity_range': (500, 5000), 'precision_range': (0.5, 2.0)},
        }
        selected_dye = 'Alexa_Fluor_647'
        selected_dye_properties = dye_properties_dict[selected_dye]
        return DNAOrigamiSimulator(1000, 'box', selected_dye_properties, augment=True, remove_corners=True,
                                   transform=pc_transforms)
    else:
        return Dataset(root_folder=root_folder, suffix=suffix, transform=pc_transforms, classes_to_use=classes,
                       data_augmentation=True, remove_part_prob=train_config['remove_part_prob'],
                       remove_corners=train_config['remove_corners'], remove_outliers=False,
                       anisotropy=train_config['anisotropy'], anisotropy_axis=train_config['anisotropy_axis'],
                       anisotropy_factor=train_config['anisotropy_factor'])


def get_model(config, train_config, device):
    return pcn.PCN(num_dense=16384, latent_dim=1024, grid_size=4, classifier=train_config['classifier'],
                   num_classes=config.dataset.number_classes).to(device)


def get_optimizers(model, train_config):
    params_recon = list(model.first_conv.parameters()) + list(model.second_conv.parameters()) + \
                   list(model.mlp.parameters()) + list(model.final_conv.parameters())
    optimizer_recon = optim.Adam(params_recon, lr=train_config['lr'], betas=(train_config['momentum'], train_config['momentum2']),
                                 weight_decay=train_config['gamma'])

    optimizer_mlp = None
    if train_config['classifier']:
        params_class = list(model.mlp1.parameters()) + list(model.bn1_mlp.parameters()) + \
                       list(model.mlp2.parameters()) + list(model.bn2_mlp.parameters()) + \
                       list(model.mlp3.parameters()) + list(model.sigmoid_mlp.parameters())
        optimizer_mlp = optim.Adam(params_class, lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-5)

    return optimizer_recon, optimizer_mlp


def get_schedulers(optimizer_recon, optimizer_mlp, train_config):
    scheduler_recon = scheduler_class = None
    if train_config['scheduler_type'] != 'None':
        if train_config['scheduler_type'] == 'StepLR':
            scheduler_recon = optim.lr_scheduler.StepLR(optimizer_recon, step_size=train_config['step_size'],
                                                        gamma=train_config['gamma'])
            if train_config['classifier']:
                scheduler_class = optim.lr_scheduler.StepLR(optimizer_mlp, step_size=20, gamma=0.1)
        else:
            scheduler_recon = optim.lr_scheduler.ReduceLROnPlateau(optimizer_recon, mode='min',
                                                                   factor=train_config['scheduler_factor'],
                                                                   patience=train_config['scheduler_patience'], verbose=True)
            if train_config['classifier']:
                scheduler_class = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mlp, mode='min',
                                                                       factor=train_config['scheduler_factor'],
                                                                       patience=train_config['scheduler_patience'], verbose=True)
        print('Scheduler activated')
    return scheduler_recon, scheduler_class


def freeze_model_layers(model):
    model.first_conv.requires_grad_(False)
    model.second_conv.requires_grad_(False)


def setup_wandb(exp_name, train_config, optimizer_recon, optimizer_mlp, fixed_alpha):
    if exp_name is not None:
        wandb.init(project='autoencoder training', name=exp_name, config=train_config)
    else:
        wandb.init(project='autoencoder training', name='PCN_tetranup', config=train_config)
    wandb.config['optimizer_recon'] = optimizer_recon
    if train_config['classifier']:
        wandb.config['optimizer_class'] = optimizer_mlp
    if fixed_alpha is not None:
        wandb.config['fixed_alpha'] = fixed_alpha


def train_and_evaluate(model, train_dataloader, val_dataloader, train_config, log_dir, device, scheduler_recon,
                       scheduler_class):
    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    alpha = train_config['initial_alpha']
    changed_lr = False

    for epoch in range(1, train_config['num_epochs'] + 1):
        wandb.log({'epoch': epoch})
        alpha = adjust_alpha(epoch, changed_lr, train_config, alpha)
        train_step(model, train_dataloader, train_config, device, alpha)
        total_cd_l1, mlp_loss = evaluate(model, val_dataloader, train_config, log_dir, epoch, device)

        if train_config['scheduler_type'] != 'None':
            if train_config['scheduler_type'] == 'StepLR':
                scheduler_recon.step()
                if train_config['classifier']:
                    scheduler_class.step()
            else:
                scheduler_recon.step(total_cd_l1)
                if train_config['classifier']:
                    scheduler_class.step(mlp_loss)

        if total_cd_l1 < best_cd_l1:
            best_cd_l1 = total_cd_l1
            best_epoch_l1 = epoch
            save_best_model(model, train_config, log_dir, epoch)

        if early_stopping(epoch, train_config, best_epoch_l1):
            break

    print('Training completed')


def adjust_alpha(epoch, changed_lr, train_config, alpha):
    if epoch == train_config['step_alpha']:
        alpha = train_config['alpha']
    if epoch == train_config['step_lr'] and not changed_lr:
        optimizer_recon.param_groups[0]['lr'] = 0.0005
        changed_lr = True
    return alpha


def train_step(model, train_dataloader, train_config, device, alpha):
    model.train()
    for step, batch in enumerate(train_dataloader):
        optimizer_recon.zero_grad()
        if train_config['classifier']:
            optimizer_mlp.zero_grad()
        pc, incomplete, gt, label = get_data_from_batch(batch, device)
        coarse, fine, loss, coarse_loss, fine_loss, mlp_loss = compute_losses(model, incomplete, gt, train_config, alpha, label)
        loss.backward()
        optimizer_recon.step()
        if train_config['classifier']:
            optimizer_mlp.step()
        log_metrics(loss, coarse_loss, fine_loss, mlp_loss)


def get_data_from_batch(batch, device):
    pc, incomplete, gt, label = batch
    incomplete = incomplete.to(device)
    gt = gt.to(device)
    label = label.to(device)
    return pc, incomplete, gt, label


def compute_losses(model, incomplete, gt, train_config, alpha, label):
    if train_config['classifier']:
        coarse, fine, latent_features, label_logit = model(incomplete)
        mlp_loss = F.binary_cross_entropy(label_logit, label)
        coarse_loss, fine_loss = losses.get_chamfer_loss(gt, coarse, fine)
        loss = alpha * (coarse_loss + fine_loss) + (1 - alpha) * mlp_loss
    else:
        coarse, fine = model(incomplete)
        mlp_loss = torch.Tensor([0])
        coarse_loss, fine_loss = losses.get_chamfer_loss(gt, coarse, fine)
        loss = alpha * (coarse_loss + fine_loss)
    return coarse, fine, loss, coarse_loss, fine_loss, mlp_loss


def log_metrics(loss, coarse_loss, fine_loss, mlp_loss):
    wandb.log({
        'loss': loss.item(),
        'coarse_loss': coarse_loss.item(),
        'fine_loss': fine_loss.item(),
        'mlp_loss': mlp_loss.item()
    })


def evaluate(model, val_dataloader, train_config, log_dir, epoch, device):
    total_cd_l1 = 0
    mlp_loss = 0
    total_loss = 0
    counter = 0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            pc, incomplete, gt, label = get_data_from_batch(batch, device)
            coarse, fine, loss, coarse_loss, fine_loss, mlp_loss = compute_losses(model, incomplete, gt, train_config, alpha, label)
            total_cd_l1 += (coarse_loss.item() + fine_loss.item())
            total_loss += loss.item()
            counter += 1
    wandb.log({'val_total_loss': total_loss / counter})
    return total_cd_l1, mlp_loss


def save_best_model(model, train_config, log_dir, epoch):
    torch.save(model.state_dict(), os.path.join(log_dir, f'best_model_epoch_{epoch}.pth'))
    print('Best model saved at epoch', epoch)


def early_stopping(epoch, train_config, best_epoch_l1):
    if epoch - best_epoch_l1 > train_config['early_stop_patience']:
        print('Early stopping at epoch', epoch)
        return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--exp_name", type=str, default=None, help="Name of the experiment")
    parser.add_argument("--alpha", type=float, default=None, help="Fixed alpha value")

    args = parser.parse_args()
    train(args.config, args.exp_name, args.alpha)
