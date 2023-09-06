import argparse

import torch
import torchvision
import torch.optim as optim
import wandb
import time
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from shutil import copyfile
from torchvision import transforms

from helpers.data import get_highest_shape
from helpers.logging import create_log_folder
from helpers.visualization import print_pc, save_plots
from model_architectures import pcn, folding_net, pointr, losses
from model_architectures.transforms import ToTensor, Padding
from model_architectures.utils import cfg_from_yaml_file, l1_cd_metric
from dataset.SMLMDataset import Dataset
from model_architectures.chamfer_distances import ChamferDistanceL2, ChamferDistanceL1
from model_architectures.pointr import validate
from chamferdist import ChamferDistance

#from chamfer_distance.chamfer_distance import ChamferDistance


#TODO: add how to install pointnet2_ops and chamfer packages in your environment
#TODO: nice to have - automatically check for cuda version and update if necessary
#TODO: add in readme that cuda version has to be updated before the chamfer libraries can be installed, ideally have also ninja build system;
#TODO: add in readme that you need the pointr and pcn chamfer dependencies - have to be installed manually from the repo in your env


wandb.login()
#cfg = cfg_from_yaml_file('configs/config.yaml')


def run_training(config_file, log_wandb=True):
    config = cfg_from_yaml_file(config_file)
    # Load models
    if config.model == 'fold':
        model = folding_net.AutoEncoder()
    elif config.model == 'pcn':
        model = pcn.PCN(config.pcn_config.coarse_dim, config.pcn_config.fine_dim)
    elif config.model == 'pointr':
        model = pointr.AdaPoinTr(config.adapointr_config)
    else:
        raise NotImplementedError('The model is not implemented yet.')

    root_folder = config.dataset.root_folder
    classes = config.dataset.classes
    suffix = config.dataset.suffix
    highest_shape = get_highest_shape(root_folder, classes)

    # Set training params
    train_config = {
        "lr": config.train.lr,
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "num_epochs": config.train.num_epochs,
        "cd_loss": ChamferDistance(),
        "early_stop_patience": config.train.early_stop_patience,
        'momentum': config.train.momentum,
        'momentum2': config.train.momentum2,
        'scheduler_type': config.train.scheduler_type,
        'gamma': float(config.train.gamma),
        'step_size': float(config.train.step_size),
        'data_augmentation': config.train.augmentation
    }

    optimizer = optim.Adam(model.parameters(), lr=train_config['lr'], betas=[train_config['momentum'], train_config['momentum2']], weight_decay = train_config['gamma'])
    if train_config['scheduler_type'] != 'None':
        if train_config['scheduler_type'] == 'StepLR':
            scheduler = StepLR(optimizer, step_size=train_config['step_size'], gamma=train_config['gamma'])
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
        print('Scheduler activated tananana')
    log_dir = create_log_folder(config.train.log_dir, config.model)

    if log_wandb:

        # Load params in wandb config
        wandb.init(project='autoencoder training', name=str(model).split('(')[0], config=train_config)
        wandb.config['optimizer'] = optimizer
        #wandb.config['scheduler'] = scheduler

    # Load data
    pc_transforms = transforms.Compose(
        [Padding(highest_shape), ToTensor()]
    )
    full_dataset = Dataset(root_folder=root_folder, suffix=suffix, transform=pc_transforms, classes_to_use=classes, data_augmentation=train_config['data_augmentation'])

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True,
                                                   num_workers=train_config['num_workers'])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False,
                                                 num_workers=train_config['num_workers'])

    # Setup device and move model to it
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    # wandb.watch(model)
    batches = int(len(train_dataset) / train_config['batch_size'] + 0.5)
    best_epoch = -1
    #min_cd_loss = 5e4
    no_improvement = 0
    current_lr = train_config['lr']
    model_saved = False
    train_step = 0
    sample_to_save_path = None

    print("The shape of the input tensor is", next(iter(train_dataloader))['pc'].shape)

    # Start training and evaluation
    print('\033[31mBegin Training...\033[0m')
    for epoch in range(1, train_config['num_epochs'] + 1):
        # training
        start = time.time()
        model.train()
        total_cd_loss_train = 0
        for i, data in enumerate(train_dataloader):
            point_clouds = data['pc']
            if config.model == 'fold' or config.model == 'pcn':
                point_clouds = point_clouds.permute(0, 2, 1)
            point_clouds = point_clouds.to(device)
            #print('This are the pointcloud shape', point_clouds.shape)
            recons = model(point_clouds)
            #print('This is the recon shape', recons[0].shape)
            if config.model == 'pcn':
                if train_step < 10000:
                    alpha = 0.01
                elif train_step < 20000:
                    alpha = 0.1
                elif train_step < 50000:
                    alpha = 0.5
                else:
                    alpha = 1.0
                loss1 = losses.cd_loss_l1(recons[0], point_clouds)
                loss2 = losses.cd_loss_l2(recons[1], point_clouds)
                ls = loss1 + alpha * loss2
            elif config.model == 'fold':
                ls = train_config['cd_loss'](point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
            elif config.model == 'pointr':
                # ls = train_config['cd_loss'](point_clouds, recons)
                sparse_loss, dense_loss = model.get_loss(recons, point_clouds)
                ls = sparse_loss + dense_loss
                #TODO: check pointr code for why does he do smth with shapenet dataset
            else:
                raise NotImplementedError('Something was wrong with the loss calculation.')
            if log_wandb:
                wandb.log({"training_step_loss": ls})

            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            total_cd_loss_train += ls
            train_step += 1

            if (i + 1) % 100 == 0:
                print('Epoch {}/{} with iteration {}/{}: CD loss is {}.'.format(epoch, train_config['num_epochs'] *
                                                                                train_config['batch_size'], i + 1,
                                                                                batches, ls))
        #mean_cd_loss_train = (total_cd_loss_train * train_config['batch_size']) / len(train_dataset)
        mean_cd_loss_train = total_cd_loss_train / (len(train_dataset)/train_config['batch_size'])
        print('\033[32mEpoch {}/{}: Best reconstructed Chamfer Distance (training) is {} in epoch {}.\033[0m'.format(
            epoch, train_config['num_epochs'], mean_cd_loss_train, best_epoch))
        if log_wandb:
            wandb.log({"training_loss": mean_cd_loss_train, "epoch": epoch})
        if epoch == 1:
            min_cd_loss = mean_cd_loss_train
        # evaluation
        model.eval()
        total_cd_loss_val = 0
        with torch.no_grad():
            for data in val_dataloader:
                if sample_to_save_path is None:
                    sample_to_save_path = data['path'][0]
                point_clouds = data['pc']
                if config.model == 'fold' or config.model == 'pcn':
                    point_clouds = point_clouds.permute(0, 2, 1)
                point_clouds = point_clouds.to(device)
                #print('This is val point_clouds shape', point_clouds.shape)
                recons = model(point_clouds)
                #print('This is val reconds0 shape', recons[0].shape)
                if config.model == 'pcn':
                    if train_step < 10000:
                        alpha = 0.01
                    elif train_step < 20000:
                        alpha = 0.1
                    elif train_step < 50000:
                        alpha = 0.5
                    else:
                        alpha = 1.0
                    loss1 = losses.cd_loss_l1(recons[0], point_clouds)
                    loss2 = losses.cd_loss_l2(recons[1], point_clouds)
                    ls = loss1 + alpha * loss2
                elif config.model == 'fold':
                    ls = train_config['cd_loss'](point_clouds.permute(0, 2, 1), recons.permute(0, 2, 1))
                elif config.model == 'pointr':
                    # ls = train_config['cd_loss'](point_clouds, recons)
                    coarse_points = recons[0]
                    dense_points = recons[-1]
                    sl1 =  losses.cd_loss_l1(coarse_points, point_clouds)
                    sl2 =  losses.cd_loss_l2(coarse_points, point_clouds)
                    dl1 =  losses.cd_loss_l1(dense_points, point_clouds)
                    dl2 =  losses.cd_loss_l2(dense_points, point_clouds)
                    if 'all' in config.train.pointr_loss:
                        ls = (sl1 + sl2 + dl1 + dl2) / 4
                    elif 'sl1' in config.train.pointr_loss:
                        ls = sl1
                    elif 'sl2' in config.train.pointr_loss:
                        ls = sl2
                    elif 'dl1' in config.train.pointr_loss:
                        ls = dl1
                    elif 'dl2' in config.train.pointr_loss:
                        ls = dl2
                    #sparse_loss, dense_loss = model.get_loss(recons, point_clouds)
                else:
                    raise NotImplementedError('Something was wrong with the loss calculation.')
                if config.model == 'fold' or config.model == 'pcn':
                    total_cd_loss_val += ls
                    if log_wandb:
                        wandb.log({"val_step_loss": ls})
                elif config.model == 'pointr':
                    total_cd_loss_val += ls
                    if log_wandb:
                        wandb.log({'val_step_loss ls': ls, 'sl1': sl1, 'sl2': sl2, 'dl1': dl1, 'dl2': dl2})

                if sample_to_save_path in data['path']:
                    index = data['path'].index(sample_to_save_path)
                    if config.model == 'pointr':
                        sample_to_save = [data['pc'][index], recons[-1][index]]
                    elif config.model == 'pcn':
                        sample_to_save = [data['pc'][index], recons[0][index]]
                    else:
                        sample_to_save = [data['pc'][index], recons.permute(0, 2, 1)[index]]
                if (i + 1) % 100 == 0:
                    print('Epoch {}/{} with iteration {}/{}: Val CD loss is {:.2f}.'.format(epoch, train_config['num_epochs'] *
                                                                                train_config['batch_size'], i + 1,
                                                                                batches, ls))

        mean_cd_loss = total_cd_loss_val / (len(val_dataset)/train_config['batch_size'])
        # calculate the mean cd loss
        # ean_cd_loss = (total_cd_loss_val * train_config['batch_size']) / len(val_dataset)

        if train_config['scheduler_type'] == 'StepLR':
            scheduler.step()
        elif train_config['scheduler_type'] == 'ReduceLROnPlateau':
            scheduler.step(mean_cd_loss)
        else:
            break
        if current_lr is not optimizer.param_groups[0]['lr'] and train_config['early_stop_patience'] - no_improvement <= 2:
            # we want to let the training run a few epochs with the new learning rate, therefore we add this if-clause
            no_improvement -= 2
            current_lr = optimizer.param_groups[0]['lr']
        if log_wandb:
            wandb.log({"val_loss": mean_cd_loss, "epoch": epoch})

        # records the best model and epoch
        if mean_cd_loss < min_cd_loss:
            no_improvement = 0
            min_cd_loss = mean_cd_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_lowest_cd_loss.pth'))
            save_plots(epoch, mean_cd_loss, sample_to_save, log_dir)
        else:
            no_improvement += 1
            if no_improvement > train_config['early_stop_patience']:
                print(
                    '\033[Early stoppage at epoch {}/{}: ; Reconstructed Chamfer Distance is {:.2f}. Minimum cd loss is {:.2f} in epoch {}.\033[0m'.format(
                        epoch, train_config['num_epochs'], mean_cd_loss, min_cd_loss, best_epoch))

                filename = str(config.train.log_dir) + str(model).split('(')[0] + '.h5'
                if log_wandb:
                    wandb.log({"epoch_stoppage": epoch})
                    wandb.save(filename)
                    copyfile(filename, wandb.run.dir)
                if log_wandb:
                    wandb.finish()
                model_saved = True
                break

        # save the model every 100 epochs
        if epoch % 100 == 0:
            torch.save(model.state_dict(), os.path.join(log_dir, 'model_epoch_{}.pth'.format(epoch)))

        end = time.time()
        cost = end - start

        print(
            '\033[32mEpoch {}/{}: Val reconstructed Chamfer Distance is {:.2f}. Minimum cd loss is {:.2f} in epoch {}.\033[0m'.format(
                epoch, train_config['num_epochs'], mean_cd_loss, min_cd_loss, best_epoch))
        print('\033[31mCost {} minutes and {} seconds\033[0m'.format(int(cost // 60), int(cost % 60)))
        if log_wandb:
            wandb.log({"lr": current_lr})
    if not model_saved and log_wandb:
        filename = str(config.train.log_dir) + str(model).split('(')[0] + '.h5'
        wandb.save(filename)
        copyfile(filename, wandb.run.dir)
    wandb.finish()


def infer_dataset(data_path, highest_shape, model_path, model_load, permute=True):
    pc_transforms = transforms.Compose([Padding(highest_shape), ToTensor()])
    infer_dataset = Dataset(data_path, '.pts', transform=pc_transforms, classes_to_use='all')
    infer_dataloader = torch.utils.data.DataLoader(infer_dataset, batch_size=1, shuffle=True, num_workers=4)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_dict = torch.load(model_path)
    model = model_load
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    for i, data in enumerate(infer_dataloader):
        point_clouds = data['pc']
        if permute:
            point_clouds = point_clouds.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        with torch.no_grad():
            recon = model(point_clouds)
        loss = ChamferDistanceL2()
        ls = loss(point_clouds.permute(0, 2, 1), recon.permute(0, 2, 1))
        print(ls, data['path'])
    return recon, point_clouds, ls


def infer(data_path, highest_shape, model_path, model_load, permute=True):
    pc_transforms = transforms.Compose([Padding(highest_shape), ToTensor()])
    infer_dataset = Dataset(data_path, '.pts', transform=pc_transforms, classes_to_use='all')
    infer_dataloader = torch.utils.data.DataLoader(infer_dataset, batch_size=1, shuffle=True, num_workers=4)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_dict = torch.load(model_path)
    model = model_load
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    reconstructions = []
    losses = []
    for i, data in enumerate(infer_dataloader):
        point_clouds = data['pc']
        if permute:
            point_clouds = point_clouds.permute(0, 2, 1)
        point_clouds = point_clouds.to(device)
        with torch.no_grad():
            recon = model(point_clouds)
        loss = ChamferDistanceL2()
        ls = loss(point_clouds.permute(0, 2, 1), recon.permute(0, 2, 1))
        #print(ls, data['path'])
        reconstructions.append(recon)
        losses.append(ls)
    return recon, point_clouds, ls


if __name__ == "__main__":
    print("\n########################")
    print("Point Cloud reconstruction")
    print("########################\n")
    parser = argparse.ArgumentParser(description='Point Cloud reconstruction')
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
    args = parser.parse_args()
    config = cfg_from_yaml_file(args.config)
    #conf = cfg_from_yaml_file('/configs/config.yaml')
    run_training(config)
    print('Done')

