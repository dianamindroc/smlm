import argparse
import torch
import torch.optim as optim
import wandb
import os
from torchvision import transforms
import numpy as np
import random
import open3d as o3d
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from joblib import  dump
import warnings

matplotlib.use('Agg')

from helpers.data import get_highest_shape
from  helpers.logging import create_log_folder
from model_architectures import pocafoldas, losses
from model_architectures.transforms import ToTensor, Padding
from model_architectures.utils import cfg_from_yaml_file, set_seed, adjust_alpha, export_ply
from dataset.SMLMDataset import Dataset
from dataset.SMLMSimulator import DNAOrigamiSimulator

warnings.filterwarnings("ignore")


def train(config, exp_name=None, wandb_log=True, fixed_alpha=None):
    set_seed(42)
    torch.backends.cudnn.benchmark = True
    config = cfg_from_yaml_file(config)
    log_dir = create_log_folder(config.train.log_dir, config.model)
    print('Loading data...')
    root_folder = config.dataset.root_folder
    suffix = config.dataset.suffix
    classes = config.dataset.classes
    highest_shape = get_highest_shape(root_folder, classes)
    pc_transforms = transforms.Compose(
        [Padding(highest_shape), ToTensor()])

    # Set
    if config.dataset.dataset_type == 'simulate':
        # Dye properties dictionary
        dye_properties_dict = {
            'Alexa_Fluor_647': {'density_range': (10, 50), 'blinking_times_range': (10, 50),
                                'intensity_range': (500, 5000),
                                'precision_range_xy': (0.5, 2.0),
                                'precision_range_z': (3.0, 5.0)},
            # ... (other dyes)
        }

        # Choose a specific dye
        selected_dye = 'Alexa_Fluor_647'
        selected_dye_properties = dye_properties_dict[selected_dye]

        full_dataset = DNAOrigamiSimulator(1000, 'box', selected_dye_properties, augment=True, remove_corners=True)
    else:
        full_dataset = Dataset(root_folder = root_folder,
                               suffix = suffix,
                               transform=pc_transforms,
                               classes_to_use=classes,
                               data_augmentation=False,
                               remove_part_prob=config.dataset.remove_part_prob,
                               remove_corners=config.dataset.remove_corners,
                               number_corners_to_remove=config.dataset.number_corners_remove,
                               remove_outliers=config.dataset.remove_outliers,
                               anisotropy=config.dataset.anisotropy,
                               anisotropy_axis=config.dataset.anisotropy_axis,
                               anisotropy_factor=config.dataset.anisotropy_factor
                               )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size,
                                                   shuffle=True,
                                                   num_workers=config.train.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False,
                                                 num_workers=config.train.num_workers)
    print("Dataset loaded!")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)

    model = pocafoldas.Pocafoldas(classifier=config.train.classifier)

    params_recon = list(model.first_conv.parameters()) + list(model.second_conv.parameters()) + \
                   list(model.mlp.parameters()) + list(model.final_conv.parameters())

    optimizer_recon = optim.Adam(params_recon, lr=config.train.lr,
                                 betas=(config.train.momentum, config.train.momentum2),
                                 weight_decay=config.train.gamma)

    if config.train.classifier:
        params_class = list(model.mlp1.parameters()) + list(model.bn1_mlp.parameters()) + \
                       list(model.mlp2.parameters()) + list(model.bn2_mlp.parameters()) + \
                       list(model.mlp3.parameters()) + list(model.sigmoid_mlp.parameters())
        optimizer_mlp = optim.Adam(params_class, lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-5)

    if config.train.scheduler_type != 'None':
        if config.train.scheduler_type == 'StepLR':
            scheduler_recon = optim.lr_scheduler.StepLR(optimizer_recon, step_size=config.train.step_size,
                                                        gamma=config.train.gamma)
            if config.train.classifier:
                scheduler_class = optim.lr_scheduler.StepLR(optimizer_mlp, step_size=20, gamma=0.1)
        else:
            scheduler_recon = optim.lr_scheduler.ReduceLROnPlateau(optimizer_recon, mode='min',
                                                                   factor=config.train.scheduler_factor,
                                                                   patience=config.train.scheduler_patience,
                                                                   verbose=True)
            if config.train.classifier:
                scheduler_class = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mlp, mode='min',
                                                                       factor=config.train.scheduler_factor,
                                                                       patience=config.train.scheduler_patience,
                                                                       verbose=True)

        print('Scheduler activated!')

    if os.path.exists(config.train.ckpt):
        model.load_state_dict(torch.load(config.train.ckpt))
        print('Model loaded from ', config.train.ckpt)
        model.first_conv.requires_grad_(False)
        model.second_conv.requires_grad_(False)

    if wandb_log:
        wandb.login()
        if exp_name is not None:
            wandb.init(project='autoencoder training', name=exp_name, config=config)
        else:
            wandb.init(project='autoencoder training', name='Experiment', config=config)
        wandb.config['optimizer_recon'] = optimizer_recon
        if config.train.classifier:
            wandb.config['optimizer_class'] = optimizer_mlp
        if fixed_alpha is not None:
            wandb.config['fixed_alpha'] = fixed_alpha

    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    train_step, val_step = 0, 0
    early_stop_counter = 0
    early_stop_patience = config.train.early_stop_patience

    # Start training
    for epoch in range(1, config.train.num_epochs + 1):
        if wandb_log:
            wandb.log({'epoch': epoch})
        if fixed_alpha is not None:
            alpha = fixed_alpha
        else:
            alpha = adjust_alpha(epoch)

        model.train()

        for i, data in enumerate(train_dataloader):
            c = data['pc']
            p = data['partial_pc']
            c_anisotropic = data['pc_anisotropic']
            if config.train.channels == 3:
                c = c[:, :, :3]
                p = p[:, :, :3]
                c_anisotropic = c_anisotropic[:, :, :3]
            label = data['label'].to(device)
            p = p.permute(0, 2, 1).to(device)
            c = c.permute(0, 2, 1).to(device) if config.train.autoencoder else c = c.to(device)
            c_anisotropic = c_anisotropic.permute(0, 2, 1).to(device)

            optimizer_recon.zero_grad()

            coarse_pred, dense_pred, _, out_classifier = model(c_anisotropic) if config.train.autoencoder else model(p)

            loss1 = losses.cd_loss_l1(coarse_pred, c)
            loss2 = losses.cd_loss_l1(dense_pred, c)
            loss = loss1 + alpha * loss2

            if wandb_log:
                wandb.log({'coarse_loss': loss1 * 1e3})
                wandb.log({'dense_loss': loss2 * 1e3})

            if config.train.classifier:
                label = label.to(torch.long)
                mlp_loss = losses.mlp_loss_function(label, out_classifier)
                optimizer_mlp.zero_grad()
                mlp_loss.backward(retain_graph=True)
                optimizer_mlp.step()

            loss.backward()
            optimizer_recon.step()
            train_step += 1

        print("Train Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, config.train.num_epochs, loss * 1e3))

        if wandb_log:
            wandb.log({'train_l1_cd': loss * 1e3})
            wandb.log({'alpha': alpha})
            export_ply(os.path.join(log_dir, '{:03d}_train_pred.ply'.format(epoch)), dense_pred[0].detach().cpu().numpy())
        torch.cuda.empty_cache()

        # Validation
        model.eval()
        total_cd_l1 = 0.0
        labels_list = []
        feature_space = []
        labels_name = []





