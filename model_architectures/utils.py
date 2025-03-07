import random

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
import yaml
from easydict import EasyDict
from types import SimpleNamespace
import open3d as o3d
from model_architectures.losses import l1_cd, l2_cd
import os

def show_point_cloud(point_cloud, axis=False):
    """visual a point cloud
    Args:
        point_cloud (np.ndarray): the coordinates of point cloud
        axis (bool, optional): Hid the coordinate of the matplotlib. Defaults to False.
    """
    ax = plt.figure().add_subplot(projection='3d')
    ax._axis3don = axis
    ax.scatter(xs=point_cloud[:, 0], ys=point_cloud[:, 1], zs=point_cloud[:, 2], s=5)
    plt.show()


def setup_seed(seed):
    """
    Set the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(point_clouds, index):
    """
    Given a batch of tensor and index, select sub-tensor.

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, N, k]
    Return:
        new_points:, indexed points data, [B, N, k, C]
    """
    device = point_clouds.device
    batch_size = point_clouds.shape[0]
    view_shape = list(index.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(index.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = point_clouds[batch_indices, index, :]
    return new_points


def knn(x, k):
    """
    K nearest neighborhood.

    Parameters
    ----------
        x: a tensor with size of (B, C, N)
        k: the number of nearest neighborhoods

    Returns
    -------
        idx: indices of the k nearest neighborhoods with size of (B, N, k)
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, 1, N), (B, N, N), (B, N, 1) -> (B, N, N)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def to_one_hots(y, categories):
    """
    Encode the labels into one-hot coding.

    :param y: labels for a batch data with size (B,)
    :param categories: total number of kinds for the label in the dataset
    :return: (B, categories)
    """
    y_ = torch.eye(categories)[y.data.cpu().numpy()]
    if y.is_cuda:
        y_ = y_.cuda()
    return y_


def get_input_shape(model):
    first_layer = model.children().__next__()

    if isinstance(first_layer, nn.Conv2d):
        return 1, first_layer.in_channels, first_layer.kernel_size[0], first_layer.kernel_size[1]
    elif isinstance(first_layer, nn.Linear):
        return 1, first_layer.in_features
    elif isinstance(first_layer, nn.RNNBase):
        return 1, first_layer.input_size
    else:
        raise ValueError("Unsupported first layer type")

def jitter_points(pc, std=0.01, clip=0.05):
    bsize = pc.size()[0]
    for i in range(bsize):
        jittered_data = pc.new(pc.size(1), 3).normal_(
            mean=0.0, std=std
        ).clamp_(-clip, clip)
        pc[i, :, 0:3] += jittered_data
    return pc


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

class LoadConfig():
    def __init__(self, path):
        with open(path, 'r') as file:
            config_dict = yaml.safe_load(file)
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, LoadConfig(value))
            else:
                setattr(self, key, value)


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)
    return config


def l2_cd_metric(pcs1, pcs2):
    dist1, dist2 = l2_cd(pcs1, pcs2)
    dist1 = torch.mean(dist1, dim=1)
    dist2 = torch.mean(dist2, dim=1)
    return torch.sum(dist1 + dist2)


def l1_cd_metric(pcs1, pcs2):
    dist1, dist2 = l1_cd(pcs1, pcs2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2

def set_seed(seed_value=42):
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # NumPy
    torch.manual_seed(seed_value)  # PyTorch
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash build-in randomization

    # If using tensorflow, add the below line
    # tf.random.set_seed(seed_value)

    # Ensures that CUDA operations are deterministic. May impact performance.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)
