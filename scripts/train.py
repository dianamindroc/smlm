import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torchvision import transforms
import wandb
import time
import os
from shutil import copyfile

from helpers.data import get_highest_shape
from helpers.logging import create_log_folder
from helpers.visualization import save_plots
from model_architectures import pcn, folding_net, losses
from model_architectures.transforms import ToTensor, Padding
from model_architectures.utils import cfg_from_yaml_file, l1_cd_metric

wandb.login()

def run_training(config_file, log_wandb=True):
    config = cfg_from_yaml_file(config_file)

    if config.model == 'fold':
        model = folding_net.AutoEncoder()
