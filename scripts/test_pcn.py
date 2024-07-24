import os
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data
import pandas as pd


from model_architectures import pcn
from dataset.ShapeNet import ShapeNet
from model_architectures.losses import l1_cd, l2_cd, f_score
from helpers.logging import create_log_folder
from model_architectures.utils import cfg_from_yaml_file, l1_cd_metric
from torchvision import transforms
from model_architectures.transforms import ToTensor, Padding
from dataset.SMLMDataset import Dataset
from helpers.data import get_highest_shape

from model_architectures.pcn_decoder import PCNDecoderOnly

ckpt = torch.load('/home/dim26fa/coding/training/logs_pcn_20240308_164948/best_l1_cd.pth')
model = pcn.PCN(classifier=False)
model.load_state_dict(ckpt)
decoder = PCNDecoderOnly(model)
final_conv_state_dict = model.final_conv.state_dict()
decoder.final_conv.load_state_dict(final_conv_state_dict)
data = np.load('saved_arrays.npz')
loaded_arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
indices = [121, 122, 123, 124, 125, 126, 127, 128, 129, 130]
indices = [29, 30, 31, 32, 33, 34, 35, 36, 40, 77]
feat_ = [elem[2] for elem in pred2]
extract = [loaded_arrays[index] for index in indices]
tensor_vectors = [torch.tensor(elem) for elem in extract]
stacked_ = torch.stack(tensor_vectors)
avg_feat = stacked_.mean(dim=0)
average_vector = avg_feat.unsqueeze(0)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
decoder.to(device)
average_vector = average_vector.to(device)
with torch.no_grad():  # Ensure no gradients are computed during inference
    coarse_output, fine_output = decoder(average_vector)
fine = fine_output[0].detach().cpu().numpy()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(fine)
o3d.visualization.draw_geometries([pcd])
logs_pcn_20240311_153830
logs_pcn_20240420_125232
pc1 = o3d.io.read_point_cloud('/home/dim26fa/coding/testing/logs_pcn_20240311_153830/121_input.ply')
o3d.visualization.draw_geometries([pc1])

def load_model(model_path, device):
    """ Load the trained model from a specified path. """
    model = pcn.PCN(num_dense=16384, latent_dim=1024, grid_size=4, classifier=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def setup_transformations(highest_shape):
    """ Set up data transformations. """
    return transforms.Compose([
        Padding(highest_shape),  # Assuming Padding and ToTensor are defined elsewhere as in training script
        ToTensor()
    ])

def load_dataset(root_folder, classes, suffix, transform, anisotropy=False):
    """ Load and return the dataset. """
    return Dataset(root_folder=root_folder, suffix=suffix, transform=transform, classes_to_use=classes, anisotropy=anisotropy)

def infer_dataset(dataset, model):
    """ Run inference on the dataset and return the results. """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    predictions = []
    model = model.to(device)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            inputs = data['partial_pc'].to(device)
            inputs = inputs.permute(0, 2, 1)  # Adjust dimensions as per model input requirement
            outputs = model(inputs)
            predictions.append(outputs)
    return predictions

def save_predictions(predictions, output_dir):
    """ Save predictions to a specified directory. """
    df = pd.DataFrame(predictions)
    df.to_csv(Path(output_dir) / "predictions.csv", index=False)


def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


def test_single_category(model, device, test_config, log_dir, save=True):
    highest_shape = get_highest_shape(test_config['root_folder'], test_config['classes_to_use'])

    pc_transforms = transforms.Compose(
    [Padding(highest_shape), ToTensor()])

    test_dataset = Dataset(root_folder=test_config['root_folder'],
                           suffix=test_config['suffix'],
                           transform=pc_transforms,
                           classes_to_use=test_config['classes_to_use'],
                           data_augmentation=True,
                           remove_part_prob = test_config['remove_part_prob'],
                           remove_corners = test_config['remove_corners'],
                           remove_outliers = False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    index = 1
    total_l1_cd, total_l2_cd = 0.0, 0.0
    feature_space = list()
    labels_list = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            c = data['pc']
            p = data['partial_pc']
            mask_complete = data['pc_mask']
            p = p.permute(0, 2, 1)
            p, c = p.to(device), c.to(device)
            _, c_, feature_global = model(p)
            print('feature_space shape:', feature_global.shape)
            total_l1_cd += l1_cd(c_, c).item()
            total_l2_cd += l2_cd(c_, c).item()
            input_pc = p[0].detach().cpu().numpy()
            input_pc = np.transpose(input_pc, (1, 0))
            output_pc = c_[0].detach().cpu().numpy()
            gt = c[0].detach().cpu().numpy()
            label = data['label'].numpy()  # Assuming 'label' is present in your dataset
            labels_list.append(label)
            if save:
                export_ply(os.path.join(log_dir, '{:03d}_input.ply'.format(index)), input_pc)
                export_ply(os.path.join(log_dir, '{:03d}_output.ply'.format(index)), output_pc)
                export_ply(os.path.join(log_dir, '{:03d}_gt.ply'.format(index)), gt)
            print('feature_global[0] shape is:', feature_global[0].shape)
            feature_space.append(feature_global[0].detach().cpu().numpy())
            index += 1
    avg_l1_cd = total_l1_cd / len(test_dataset)
    avg_l2_cd = total_l2_cd / len(test_dataset)
#    avg_f_score = total_f_score / len(test_dataset)
    labels_array = np.concatenate(labels_list, axis=0)
    return avg_l1_cd, avg_l2_cd, feature_space, labels_array


def test(config, save=True):
    features = list()
    config = cfg_from_yaml_file(config)
    test_config = {
        'root_folder': config.dataset.root_folder,
        'classes_to_use': config.dataset.classes,
        'save': config.test.save,
        'ckpt_path': config.test.ckpt_path,
        'suffix': config.dataset.suffix,
        'remove_part_prob': config.dataset.remove_part_prob,
        'remove_corners': config.dataset.remove_corners}
    if save:
        log_dir = create_log_folder(config.test.log_dir, config.model)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load pretrained model
    model = pcn.PCN(16384, 1024, 4).to(device)
    state_dict = torch.load(test_config['ckpt_path'])
    if 'mlp3.weight' in state_dict.keys():
        keys_to_remove = [
            "mlp1.weight", "mlp1.bias",
            "bn1_mlp.weight", "bn1_mlp.bias", "bn1_mlp.running_mean", "bn1_mlp.running_var",
            "bn1_mlp.num_batches_tracked",
            "mlp2.weight", "mlp2.bias",
            "bn2_mlp.weight", "bn2_mlp.bias", "bn2_mlp.running_mean", "bn2_mlp.running_var",
            "bn2_mlp.num_batches_tracked",
            "mlp3.weight", "mlp3.bias"
        ]

        # Assuming `my_dict` is your dictionary-like structure
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]
    model.load_state_dict(state_dict)
    model.eval()

    print('\033[33m{:20s}{:20s}{:20s}\033[0m'.format('Category', 'L1_CD(1e-3)', 'L2_CD(1e-4)'))
    print('\033[33m{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------'))

    l1_cds, l2_cds = list(), list()
    avg_l1_cd, avg_l2_cd, feature_space, labels_array = test_single_category(model, device, test_config, log_dir, save)
    print('\033[32m{:20s}{:20.4f}{:20.4f}\033[0m'.format('All', avg_l1_cd * 1e3, avg_l2_cd * 1e4))
    return avg_l1_cd, avg_l2_cd, feature_space, labels_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--exp_name', type=str, help='Tag of experiment')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--ckpt_path', type=str, help='The path of pretrained model.')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=6, help='Num workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for testing')
    parser.add_argument('--save', type=bool, default=False, help='Saving test result')
    parser.add_argument('--novel', type=bool, default=False, help='unseen categories for testing')
    params = parser.parse_args()
    test(params)

