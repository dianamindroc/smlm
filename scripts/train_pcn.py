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
from joblib import dump
from scipy.spatial.transform import Rotation as R

matplotlib.use('Agg')

from helpers.data import get_highest_shape
from helpers.logging import create_log_folder
from model_architectures import pcn, losses
from model_architectures.transforms import ToTensor, Padding
from model_architectures.utils import cfg_from_yaml_file, set_seed
from dataset.SMLMDataset import Dataset
from dataset.ShapeNet import ShapeNet
from dataset.SMLMSimulator import DNAOrigamiSimulator
import warnings

warnings.filterwarnings("ignore")


def train(config, exp_name=None, fixed_alpha=None):
    global total_cd_l1
    set_seed(42)
    wandb.login()
    torch.backends.cudnn.benchmark = True
    config = cfg_from_yaml_file(config)
    train_config = {
        "lr": config.train.lr,
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "num_epochs": config.train.num_epochs,
        "loss": config.train.loss,
        "early_stop_patience": config.train.early_stop_patience,
        'momentum': config.train.momentum,
        'momentum2': config.train.momentum2,
        'scheduler_type': config.train.scheduler_type,
        'scheduler_factor': config.train.scheduler_factor,
        'scheduler_patience': config.train.scheduler_patience,
        'gamma': config.train.gamma,
        'step_size': config.train.step_size,
        'remove_part_prob': config.dataset.remove_part_prob,
        'dataset': config.dataset.dataset_type,
        'which_dataset': config.dataset.root_folder,
        'remove_corners': config.dataset.remove_corners,
        'num_corners_remove': config.dataset.number_corners_remove,
        'anisotropy': config.dataset.anisotropy,
        'anisotropy_factor': config.dataset.anisotropy_factor,
        'anisotropy_axis': config.dataset.anisotropy_axis,
        'classifier': config.train.classifier,
        'ckpt': config.train.ckpt,
        'autoencoder': config.train.autoencoder,
        'lambda_reg': config.train.lambda_reg,
        'alpha_density': config.train.alpha_density,
        'beta_sparsity': config.train.beta_sparsity,
        'channels': config.train.channels,
        'remove_outliers': config.dataset.remove_outliers,
        'gml_sigma': config.train.gml_sigma,
        'grid_size': config.train.grid_size,
        'min_lr': config.train.min_lr,
        'latent_dim': config.train.latent_dim,
        'num_dense': config.train.num_dense
    }
    log_dir = create_log_folder(config.train.log_dir, config.model)
    print('Loading Data...')
    root_folder = config.dataset.root_folder
    classes = config.dataset.classes
    suffix = config.dataset.suffix
    highest_shape = get_highest_shape(root_folder, classes)
    pc_transforms = transforms.Compose(
        [Padding(highest_shape), ToTensor()])

    if train_config['dataset'] == 'shapenet':
        train_dataset = ShapeNet(root_folder, 'train', classes)
        val_dataset = ShapeNet(root_folder, 'valid', classes)
    elif train_config['dataset'] == 'simulate':
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

        full_dataset = DNAOrigamiSimulator(1000, 'box', selected_dye_properties, augment=True, remove_corners=True,
                                           transform=pc_transforms)
    else:
        full_dataset = Dataset(root_folder=root_folder,
                               suffix=suffix,
                               transform=pc_transforms,
                               classes_to_use=classes,
                               data_augmentation=False,
                               remove_part_prob=train_config['remove_part_prob'],
                               remove_corners=train_config['remove_corners'],
                               remove_outliers=train_config['remove_outliers'],
                               anisotropy=train_config['anisotropy'],
                               anisotropy_axis=train_config['anisotropy_axis'],
                               anisotropy_factor=train_config['anisotropy_factor'],
                               number_corners_to_remove=train_config['num_corners_remove'])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True,
                                                   num_workers=train_config['num_workers'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False,
                                                 num_workers=train_config['num_workers'])
    print("Dataset loaded!")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)

    # model
    model = pcn.PCN(num_dense=train_config['num_dense'], latent_dim=train_config['latent_dim'], grid_size=train_config['grid_size'],
                    classifier=train_config['classifier'],
                    num_classes=config.dataset.number_classes, channels=train_config['channels']).to(device)
    # optimizer_recon
    #optimizer_recon = optim.Adam(model.parameters(), lr=train_config['lr'], betas=(0.9, 0.999))
    #lr_schedual = optim.lr_scheduler.StepLR(optimizer_recon, step_size=50, gamma=0.7)
    params_recon = list(model.first_conv.parameters()) + list(model.second_conv.parameters()) + \
                   list(model.mlp.parameters()) + list(model.final_conv.parameters())
    optimizer_recon = optim.Adam(params_recon, lr=train_config['lr'],
                                 betas=(train_config['momentum'], train_config['momentum2']),
                                 weight_decay=train_config['gamma'], )
    if train_config['classifier']:
        params_class = list(model.mlp1.parameters()) + list(model.bn1_mlp.parameters()) + \
                       list(model.mlp2.parameters()) + list(model.bn2_mlp.parameters()) + \
                       list(model.mlp3.parameters()) + list(model.sigmoid_mlp.parameters())
        optimizer_mlp = optim.Adam(params_class, lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-5)

    if train_config['scheduler_type'] != 'None':
        if train_config['scheduler_type'] == 'StepLR':
            scheduler_recon = optim.lr_scheduler.StepLR(optimizer_recon, step_size=train_config['step_size'],
                                                        gamma=train_config['gamma'])
            if train_config['classifier']:
                scheduler_class = optim.lr_scheduler.StepLR(optimizer_mlp, step_size=20, gamma=0.1)
        else:
            scheduler_recon = optim.lr_scheduler.ReduceLROnPlateau(optimizer_recon, mode='min',
                                                                   factor=train_config['scheduler_factor'],
                                                                   patience=train_config['scheduler_patience'],
                                                                   verbose=True)
            if train_config['classifier']:
                scheduler_class = optim.lr_scheduler.ReduceLROnPlateau(optimizer_mlp, mode='min',
                                                                       factor=train_config['scheduler_factor'],
                                                                       patience=train_config['scheduler_patience'],
                                                                       verbose=True)
        print('Scheduler activated')

    if os.path.exists(train_config['ckpt']):
        model.load_state_dict(torch.load(train_config['ckpt']))
        print('Model loaded from ', train_config['ckpt'])
        model.first_conv.requires_grad_(False)
        model.second_conv.requires_grad_(False)
        #model.mlp.requires_grad_(False)
        #print('Frozen all but last layer')

    if exp_name is not None:
        wandb.init(project='autoencoder training', name=exp_name, config=train_config)
    else:
        wandb.init(project='autoencoder training', name='PCN_tetranup', config=train_config)
    wandb.config['optimizer_recon'] = optimizer_recon
    if train_config['classifier']:
        wandb.config['optimizer_class'] = optimizer_mlp
    if fixed_alpha is not None:
        wandb.config['fixed_alpha'] = fixed_alpha

    # training
    best_cd_l1 = 1e8
    best_epoch_l1 = -1
    train_step, val_step = 0, 0
    alpha = config.train.initial_alpha
    loss1_improvement_threshold = config.train.loss1_improvement_threshold  # Define your own threshold
    loss1_history = []
    min_improvement_epochs = config.train.min_improvement_epochs  # Number of epochs to look back for improvement
    changed_lr = False
    early_stop_counter = 0
    early_stop_patience = config.train.early_stop_patience
    for epoch in range(1, train_config['num_epochs'] + 1):
        wandb.log({'epoch': epoch})
        # hyperparameter alpha
        if fixed_alpha is not None:
            alpha = fixed_alpha
        else:
            alpha = adjust_alpha(epoch)

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
            c_anisotropic = data['pc_anisotropic']
            #rot = data['rotation']
            if train_config['channels'] == 3:
                c = c[:, :, :3]
                p = p[:, :, :3]
                c_anisotropic = c_anisotropic[:, :, :3]
            #mask_complete = data['pc_mask']
            label = data['label']
            if train_config['autoencoder']:
                c_per = c.permute(0, 2, 1)
                c_per = c_per.to(device)
                c_anisotropic = c_anisotropic.permute(0, 2, 1)
                c_anisotropic = c_anisotropic.to(device)
            p = p.permute(0, 2, 1)
            p, c, label = p.to(device), c.to(device), label.to(device)
            optimizer_recon.zero_grad()

            # forward propagation
            if train_config['autoencoder']:
                #coarse_pred, dense_pred, _, out_classifier = model(c_per)
                coarse_pred, dense_pred, _, out_classifier = model(c_anisotropic)
            else:
                #coarse_pred, dense_pred, _, out_classifier = model(p)
                coarse_pred, dense_pred, _, out_classifier = model(p)

            # try out rotations information
            # rotated_reconstructions = []
            # for i, rec in enumerate(dense_pred):
            #     if dense_pred[i] is not None:  # Check if we have rotation for this sample
            #         # Create rotation object
            #         rotation = R.from_quat(rot[i].detach().cpu().numpy())  # Convert from tensor if needed
            #         # Apply rotation to reconstructed points
            #         rotated_rec = torch.tensor(rotation.apply(rec.detach().cpu().numpy()),
            #                                                   dtype=torch.float32,
            #                                                   device=rec.device,
            #                                                   requires_grad=True)
            #         rotated_reconstructions.append(rotated_rec)  # Back to tensor
            #     else:
            #         rotated_reconstructions.append(rec)
            #
            # rotated_reconstructions = torch.stack(rotated_reconstructions).to(device)

            # loss function
            if train_config['loss'] == 'dcd':
                # Compute DCD loss for both coarse and dense predictions
                dcd_loss1, cd_p1, cd_t1 = losses.calc_dcd(coarse_pred, c)
                dcd_loss2, cd_p2, cd_t2 = losses.calc_dcd(dense_pred, c)
                loss = dcd_loss1.mean() + alpha * dcd_loss2.mean()
            elif train_config['loss'] == 'adapted':
                from model_architectures.losses import _compute_density, _with_density, _sparsity_regularization, \
                    safe_normalize
                loss1 = losses.cd_loss_l1(coarse_pred, c)
                loss2 = losses.cd_loss_l1(dense_pred, c)
                cd_loss = loss1 + alpha * loss2
                density_pred = _compute_density(dense_pred)
                density_gt = _compute_density(c)
                density_loss = _with_density(dense_pred, c, density_pred, density_gt)
                #sparsity_loss = torch.tensor(_sparsity_regularization(dense_pred, train_config['lambda_reg']))
                loss = cd_loss + train_config[
                    'alpha_density'] * density_loss  # + train_config['beta_sparsity'] * sparsity_loss
                #print(f'CD Loss: {cd_loss}, Density Loss: {density_loss}, Sparsity Loss: {sparsity_loss}')
            elif train_config['loss'] == 'own_dcd':
                from model_architectures.losses import DensityAwareChamferLoss
                density_aware_chamfer_loss = DensityAwareChamferLoss()
                loss = density_aware_chamfer_loss(dense_pred, c)
            elif train_config['loss'] == 'gml':
                from model_architectures.losses import GMMLoss
                gml_loss = GMMLoss()
                loss1 = gml_loss(coarse_pred, c)
                loss2 = losses.cd_loss_l1(dense_pred, c)
                loss = loss1 + alpha * loss2
            else:
                loss1 = losses.cd_loss_l1(coarse_pred, c)
                loss2 = losses.cd_loss_l1(dense_pred, c)
                # loss1 = losses.cd_loss_l1(rotated_reconstructions, c)
                # loss2 = losses.cd_loss_l1(rotated_reconstructions, c)
                loss = loss1 + alpha * loss2
                wandb.log({'coarse_loss': loss1 * 1e3})
                wandb.log({'dense_loss': loss2 * 1e3})

            if train_config['classifier']:
                label = label.to(torch.long)
                mlp_loss = losses.mlp_loss_function(label, out_classifier)
                optimizer_mlp.zero_grad()
                mlp_loss.backward(retain_graph=True)
                optimizer_mlp.step()
                #print(F.softmax(out_classifier, dim=0))

            # back propagation
            loss.backward()
            optimizer_recon.step()
            train_step += 1

        print("Train Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, train_config['num_epochs'],
                                                                                 loss * 1e3))

        wandb.log({'train_l1_cd': loss * 1e3})
        wandb.log({'alpha': alpha})
        export_ply(os.path.join(log_dir, '{:03d}_train_pred.ply'.format(epoch)), dense_pred[0].detach().cpu().numpy())
        if train_config['loss'] == 'adapted':
            del p, c, coarse_pred, dense_pred, cd_loss, density_pred, density_gt, loss, density_loss  #, sparsity_loss
        torch.cuda.empty_cache()
        #export_ply(os.path.join(log_dir, '{:03d}.ply'.format(i)), np.transpose(dense_pred[0].detach().cpu().numpy(), (1,0)))
        #lr_schedual.step()

        # evaluation
        model.eval()
        total_cd_l1 = 0.0
        labels_list = []
        corner_label_list = []
        corner_name = []
        feature_space = []
        labels_names = []
        with torch.no_grad():
            rand_iter = random.randint(0, len(val_dataloader) - 1)
            # for visualization
            #### ShapeNet dataset
            #for i, (p, c) in enumerate(val_dataloader):
            #    p = p.permute(0, 2, 1)
            #    p, c = p.to(device), c.to(device)

            #### SMLMDataset
            for i, data in enumerate(val_dataloader):
                c = data['pc']
                p = data['partial_pc']
                c_anisotropic = data['pc_anisotropic']
                #rot = data['rotation']
                filenames = [os.path.basename(path) for path in data['path']]
                if train_config['channels'] == 3:
                    c = c[:, :, :3]
                    p = p[:, :, :3]
                    c_anisotropic = c_anisotropic[:, :, :3]
                mask_complete = data['pc_mask']
                label = data['label']
                corner_label = data['corner_label']
                # filename = os.path.basename(data['path'])
                if train_config['autoencoder']:
                    c_per = c.permute(0, 2, 1)
                    c_per = c_per.to(device)
                    c_anisotropic = c_anisotropic.permute(0, 2, 1)
                    c_anisotropic = c_anisotropic.to(device)
                p = p.permute(0, 2, 1)
                p, c, label, corner_label = p.to(device), c.to(device), label.to(device), corner_label.to(device)
                if train_config['autoencoder']:
                    #coarse_pred, dense_pred, features, out_classifier = model(c_per)
                    coarse_pred, dense_pred, features, out_classifier = model(c_anisotropic)
                else:
                    #coarse_pred, dense_pred, features, out_classifier = model(p)
                    coarse_pred, dense_pred, features, out_classifier = model(p)

                # try out rotations information
                #rotated_reconstructions = []
                #for b, rec in enumerate(dense_pred):
                #    if dense_pred[b] is not None:  # Check if we have rotation for this sample
                        # Create rotation object
                #        rotation = R.from_quat(rot[b].detach().cpu().numpy())  # Convert from tensor if needed
                        # Apply rotation to reconstructed points
                #        rotated_rec = torch.tensor(rotation.apply(rec.detach().cpu().numpy()),
                #                                              dtype=torch.float32,
                #                                              device=rec.device,
                #                                              requires_grad=True)
                #        rotated_reconstructions.append(rotated_rec)  # Back to tensor
                #    else:
                #        rotated_reconstructions.append(rec)

                #rotated_reconstructions = torch.stack(rotated_reconstructions).to(device)

                labels_list.append(data['label'].numpy())
                corner_label_list.append(data['corner_label'].numpy())
                corner_name.append(data['corner_label_name'])
                labels_names.append(data['label_name'])
                feature_space.extend(features.detach().cpu().numpy())
                cd_loss = losses.l1_cd(dense_pred, c).item()
                total_cd_l1 += cd_loss
                mlp_loss = losses.mlp_loss_function(label, out_classifier)
                if i == rand_iter:
                    j = random.randint(0, len(filenames) - 1)
                    if train_config['autoencoder']:
                        input_pc = c_anisotropic[j].detach().cpu().numpy()
                    else:
                        input_pc = p[j].detach().cpu().numpy()
                    input_pc = np.transpose(input_pc, (1, 0))
                    output_pc = dense_pred[j].detach().cpu().numpy()
                    #rotated_output = rotated_reconstructions[j].detach().cpu().numpy()
                    gt = c[j].detach().cpu().numpy()
                    coarse = coarse_pred[j].detach().cpu().numpy()

                    sample_filename = filenames[j]

                    # Export PLY files
                    export_ply(os.path.join(log_dir, f'{epoch:03d}_input_{sample_filename}.ply'), input_pc)
                    export_ply(os.path.join(log_dir, f'{epoch:03d}_output_{sample_filename}.ply'), output_pc)
                    export_ply(os.path.join(log_dir, f'{epoch:03d}_gt_{sample_filename}.ply'), gt)
                    export_ply(os.path.join(log_dir, f'{epoch:03d}_coarse_{sample_filename}.ply'), coarse)
                    #export_ply(os.path.join(log_dir, f'{epoch:03d}_rotatedoutput_{sample_filename}.ply'), rotated_output)

                    # Log to wandb
                    wandb.log({
                        "point_cloud_val": [
                            wandb.Object3D(gt[:, :3]),
                            wandb.Object3D(input_pc[:, :3]),
                            wandb.Object3D(output_pc[:, :3])
                        ],
                        "logged_sample_filename": sample_filename,
                        "epoch": epoch
                    })


            min_lr = train_config['min_lr']  # This allows 4 zeros after the decimal point, but not 5

            total_cd_l1 /= len(val_dataset)
            if train_config['scheduler_type'] != 'None':
                if train_config['scheduler_type'] == 'StepLR':
                    scheduler_recon.step()
                    if train_config['classifier']:
                        scheduler_class.step()
                else:
                    prev_lr = optimizer_recon.param_groups[0]['lr']

                    # Check if the current learning rate is above the minimum before stepping
                    if prev_lr > min_lr:
                        scheduler_recon.step(total_cd_l1)
                        if train_config['classifier']:
                            scheduler_class.step(mlp_loss)

                        current_lr = optimizer_recon.param_groups[0]['lr']

                        # Ensure the new learning rate is not below the minimum
                        if current_lr < min_lr:
                            for param_group in optimizer_recon.param_groups:
                                param_group['lr'] = min_lr
                            current_lr = min_lr

                        if prev_lr != current_lr:
                            wandb.log({"lr": current_lr})
                            print(f"Learning rate updated to: {current_lr:.5f}")
                    else:
                        print(f"Learning rate already at minimum: {prev_lr:.5f}")
            val_step += 1

            print(
                "Validate Epoch [{:03d}/{:03d}]: L1 Chamfer Distance = {:.6f}".format(epoch, train_config['num_epochs'],
                                                                                      total_cd_l1 * 1e3))
            feature_array = np.vstack(feature_space)
            labels_array = np.concatenate(labels_list)
            corner_array = np.concatenate(corner_label_list)
            labels_names = np.concatenate(labels_names)
            corner_name = np.concatenate(corner_name)
            plot_tsne(feature_array, labels_array, labels_names, epoch, log_dir)
            plot_tsne(feature_array, corner_array, corner_name, 111, log_dir)
            pca = plot_pca(feature_array, labels_array, labels_names, epoch, log_dir)
            save_pca_model(pca, log_dir, epoch)
            wandb.log({'val_l1_cd': total_cd_l1 * 1e3})
            wandb.log({'mlp_loss': mlp_loss * 1e3})

        if total_cd_l1 < best_cd_l1:
            best_epoch_l1 = epoch
            best_cd_l1 = total_cd_l1
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_l1_cd.pth'))
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f'Early stopping triggered. No improvement for {early_stop_patience} epochs.')
            break

    torch.cuda.empty_cache()
    print('Best l1 cd model in epoch {}, the minimum l1 cd is {}'.format(best_epoch_l1, best_cd_l1 * 1e3))


def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


def plot_tsne(features, labels, labels_names, epoch, log_dir):
    #color_map = {0: '#009999', 1: '#FFB266'}
    label_to_name = {label: name for label, name in zip(np.unique(labels), np.unique(labels_names))}

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(features)
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('nipy_spectral', len(unique_labels))
    for i, label in enumerate(unique_labels):
        # Find points in this cluster
        idx = labels == label
        label_name = label_to_name[label]
        # Plot these points with a unique color and label
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=[colors(i)], label=f'Cluster {label_name}')
    plt.title(f't-SNE visualization (Epoch {epoch})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()

    # Save the plot with a specific filename format
    plt.savefig(os.path.join(log_dir, '{:03d}_tsne.png'.format(epoch)))
    # Close the figure to free memory
    plt.close()


def plot_pca(features, labels, labels_names, epoch, log_dir, pca=None):
    # Create a mapping from label to name
    label_to_name = {label: name for label, name in zip(np.unique(labels), np.unique(labels_names))}

    if pca is None:
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(features)
    else:
        pca_results = pca.transform(features)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(features)

    # Plot the results
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('nipy_spectral', len(unique_labels))

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(unique_labels):
        idx = labels == label
        label_name = label_to_name[label]
        plt.scatter(pca_results[idx, 0], pca_results[idx, 1], c=[colors(i)], label=f'Cluster {label_name}')

    plt.title(f'PCA visualization (Epoch {epoch})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()

    # Save the plot
    plt.savefig(f'{log_dir}/pca_visualization_epoch_{epoch}.png')
    plt.close()

    return pca


def adjust_alpha(epoch):
    if epoch < 120:
        alpha = 0.001  # was 0.01
    elif epoch < 200:
        alpha = 0.005  # was 0.1, 0.02
    elif epoch < 300:
        alpha = 0.01  # was 0.5, 0.05
    else:
        alpha = 0.02  # was 1, 0.1
    return alpha


def save_pca_model(pca, log_dir, epoch):
    model_path = os.path.join(log_dir, f'pca_model_epoch_{epoch}.joblib')
    dump(pca, model_path)
    print(f"PCA model saved to {model_path}")



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
