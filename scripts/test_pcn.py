import os
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import torch.utils.data as Data
import pandas as pd
import torch.nn.functional as F

from model_architectures.pcn import PCN
from dataset.ShapeNet import ShapeNet
from model_architectures.losses import l1_cd, l2_cd, f_score
from helpers.logging import create_log_folder
from model_architectures.utils import cfg_from_yaml_file, l1_cd_metric
from torchvision import transforms
from model_architectures.transforms import ToTensor, Padding
from dataset.SMLMDataset import Dataset
from helpers.data import get_highest_shape
from joblib import dump, load
from model_architectures.pcn_decoder import PCNDecoderOnly

ckpt = torch.load('/home/dim26fa/coding/training/logs_pcn_20241009_225619/best_l1_cd.pth')
model = PCN(classifier=True)
model_state_dict = model.state_dict()
# Filter out layers with size mismatches
filtered_checkpoint = {k: v for k, v in ckpt.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
model_state_dict.update(filtered_checkpoint)
model.load_state_dict(model_state_dict)
model.load_state_dict(ckpt, strict=False)
decoder = PCNDecoderOnly(model)
final_conv_state_dict = model.final_conv.state_dict()
decoder.final_conv.load_state_dict(final_conv_state_dict)
data = np.load('saved_arrays.npz')
loaded_arrays = [data[f'arr_{i}'] for i in range(len(data.files))]
indices = [50, 60]
indices = [1, 66, 23, 54]
feat_ = [elem[2] for elem in pred2]
extract = [feature_space[index] for index in indices]
extract_exp = [feat_space_exp[index] for index in indices]
tensor_vectors = [torch.tensor(elem) for elem in extract]
tensor_vectors = [torch.tensor(elem) for elem in feature_space]
tensor_vectors = [torch.tensor(elem) for elem in extract_exp]
stacked_ = torch.stack(tensor_vectors)
# --------------------------------------------------
avg_feat = stacked_.mean(dim=0)
average_vector = avg_feat.unsqueeze(0)
# attention based averaging?------------------------
attention = AttentionMechanism(1024)
attention.eval()
with torch.no_grad():
    avg_embedding_attention = attention(stacked_)
average_vector = avg_embedding_attention.unsqueeze(0)
# PointNet averaging? ------------------------------
pointnet = PointNetForEmbeddings(input_dim=1024, hidden_dim=512, output_dim=1024)
global_feature = pointnet(stacked_)
average_vector = global_feature.unsqueeze(0)
# --------------------------------------------------
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
    return Dataset(root_folder=root_folder, suffix=suffix, transform=transform, classes_to_use=classes,
                   anisotropy=anisotropy)


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


def load_pca_model(log_dir, epoch):
    model_path = os.path.join(log_dir, f'pca_model_epoch_{epoch}.joblib')
    if os.path.exists(model_path):
        pca = load(model_path)
        print(f"PCA model loaded from {model_path}")
        return pca
    else:
        print(f"No PCA model found at {model_path}")
        return None


def test_single_category(model, device, test_config, log_dir, save=True):
    highest_shape = get_highest_shape(test_config['root_folder'], test_config['classes_to_use'])

    pc_transforms = transforms.Compose(
        [Padding(highest_shape), ToTensor()])

    test_dataset = Dataset(root_folder=test_config['root_folder'],
                           suffix=test_config['suffix'],
                           transform=pc_transforms,
                           classes_to_use=test_config['classes_to_use'],
                           data_augmentation=False,
                           remove_part_prob=test_config['remove_part_prob'],
                           remove_corners=test_config['remove_corners'],
                           remove_outliers=test_config['remove_outliers'],
                           anisotropy=test_config['anisotropy'],
                           anisotropy_axis=test_config['anisotropy_axis'],
                           anisotropy_factor=test_config['anisotropy_factor'],
                           number_corners_to_remove=test_config['num_corners_remove'])
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    index = 1
    total_l1_cd, total_l2_cd = 0.0, 0.0
    feature_space = list()
    labels_list = []
    corner_label_list = []
    corner_name = []
    l1cd_list = []
    l2cd_list = []
    with torch.no_grad():
        out_classifier_list = []
        labels_names = []
        for i, data in enumerate(test_dataloader):
            c = data['pc']
            c = c[:, :, :3]
            p = data['partial_pc']
            p = p[:, :, :3]
            c_anisotropic = data['pc_anisotropic']
            c_anisotropic = c_anisotropic[:, :, :3]
            mask_complete = data['pc_mask']
            filename = os.path.basename(data['path'][0])
            label = data['label']
            corner_label = data['corner_label']
            p = p.permute(0, 2, 1)
            p, c, label, corner_label = p.to(device), c.to(device), label.to(device), corner_label.to(device)
            _, c_, feature_global, out_classifier = model(p)
            print('feature_space shape:', feature_global.shape)
            total_l1_cd += l1_cd(c_, c).item()
            l1cd_list.append(l1_cd(c_, c).item())
            total_l2_cd += l2_cd(c_, c).item()
            l2cd_list.append(l2_cd(c_, c).item())
            input_pc = p[0].detach().cpu().numpy()
            input_pc = np.transpose(input_pc, (1, 0))
            output_pc = c_[0].detach().cpu().numpy()
            gt = c[0].detach().cpu().numpy()
            label = data['label'].numpy()  # Assuming 'label' is present in your dataset
            labels_list.append(label)
            labels_names.append(data['label_name'])
            corner_label_list.append(data['corner_label'].numpy())
            corner_name.append(data['corner_label_name'])
            if save:
                export_ply(os.path.join(log_dir, f'{index:03d}_input_{filename}.ply'), input_pc)
                export_ply(os.path.join(log_dir, f'{index:03d}_output{filename}.ply'), output_pc)
                export_ply(os.path.join(log_dir, f'{index:03d}_gt{filename}.ply'), gt)
            print('feature_global[0] shape is:', feature_global[0].shape)
            feature_space.append(feature_global[0].detach().cpu().numpy())
            out_classifier = F.softmax(out_classifier, dim=1)
            out_classifier_list.append(out_classifier.detach().cpu().numpy())

            index += 1
    avg_l1_cd = total_l1_cd / len(test_dataset)
    avg_l2_cd = total_l2_cd / len(test_dataset)
    #    avg_f_score = total_f_score / len(test_dataset)
    labels_array = np.concatenate(labels_list, axis=0)
    labels_names = np.concatenate(labels_names)
    corner_array = np.concatenate(corner_label_list)
    corner_name = np.concatenate(corner_name)
    out_classifier_array = np.concatenate(out_classifier_list, axis=0)
    predicted_class_array = np.argmax(out_classifier_array, axis=1)
    return avg_l1_cd, avg_l2_cd, feature_space, labels_array, labels_names, corner_array, corner_name, out_classifier_array, predicted_class_array


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
        'remove_corners': config.dataset.remove_corners,
        'anisotropy': config.dataset.anisotropy,
        'anisotropy_factor': config.dataset.anisotropy_factor,
        'anisotropy_axis': config.dataset.anisotropy_axis,
        'remove_outliers': config.dataset.remove_outliers,
        'classifier': config.test.classifier,
        'num_corners_remove': config.dataset.number_corners_remove}
    if save:
        log_dir = create_log_folder(config.test.log_dir, config.model)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load pretrained model
    model = PCN(16384, 1024, 4, classifier=test_config['classifier']).to(device)
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
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print('\033[33m{:20s}{:20s}{:20s}\033[0m'.format('Category', 'L1_CD(1e-3)', 'L2_CD(1e-4)'))
    print('\033[33m{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------'))

    l1_cds, l2_cds = list(), list()
    avg_l1_cd, avg_l2_cd, feature_space, labels_array, labels_names, corner_array, corner_name, out_classifier_array, pred_labels_array = test_single_category(model, device, test_config, log_dir, save)
    print('\033[32m{:20s}{:20.4f}{:20.4f}\033[0m'.format('All', avg_l1_cd * 1e3, avg_l2_cd * 1e4))
    return avg_l1_cd, avg_l2_cd, feature_space, labels_array, labels_names, corner_array, corner_name, out_classifier_array, pred_labels_array

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionMechanism, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, embeddings):
        # Compute scores for each embedding
        scores = torch.matmul(embeddings, self.attention_weights)
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=0)
        # Compute the weighted average of embeddings
        return torch.sum(weights[:, None] * embeddings, dim=0)


class AttentionAggregator(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionAggregator, self).__init__()
        self.attention = nn.Linear(embedding_dim, 1)

    def forward(self, embeddings):
        # embeddings shape: (num_embeddings, embedding_dim)
        attention_weights = F.softmax(self.attention(embeddings), dim=0)
        aggregated_embedding = torch.sum(attention_weights * embeddings, dim=0)
        return aggregated_embedding


class MultiHeadAttentionAggregator(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1):
        super(MultiHeadAttentionAggregator, self).__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, embeddings):
        # embeddings shape: (num_embeddings, embedding_dim)
        embeddings = embeddings.unsqueeze(1)  # Add sequence length dimension

        # Self-attention
        attn_output, _ = self.attention(embeddings, embeddings, embeddings)
        attn_output = attn_output.squeeze(1)

        # Add & Norm
        normalized = self.layer_norm(embeddings.squeeze(1) + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(normalized)

        # Final Add & Norm
        output = self.final_layer_norm(normalized + ffn_output)

        return output.mean(dim=0)  # Average across embeddings

def aggregate_embeddings(embeddings, aggregator, weights=None):
    """
    Aggregate a list of embeddings using the advanced aggregator.

    Args:
    embeddings (List[torch.Tensor]): List of embedding tensors
    aggregator (MultiHeadAttentionAggregator): The aggregator module
    weights (torch.Tensor, optional): Optional weights for each embedding

    Returns:
    torch.Tensor: The aggregated embedding
    """
    stacked_embeddings = torch.stack(embeddings)

    if weights is not None:
        assert len(weights) == len(embeddings), "Number of weights must match number of embeddings"
        weights = weights.unsqueeze(1)  # Add dimension for broadcasting
        stacked_embeddings = stacked_embeddings * weights

    return aggregator(stacked_embeddings)

class PointNetForEmbeddings(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PointNetForEmbeddings, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Ensure output_dim matches input_dim
        )

    def forward(self, x):
        # x shape: (N, input_dim)
        x = self.mlp(x)  # Apply shared MLP
        # x shape: (N, output_dim)
        x = torch.max(x, dim=0, keepdim=True)[0]  # Apply max pooling over the points (N)
        return x.squeeze(0)  # Remove the extra dimension added by keepdim=True


def simple_aggregate_embeddings(embeddings):
    """
    Aggregate point cloud embeddings using element-wise median.

    Args:
    embeddings (List[torch.Tensor]): List of embedding tensors, each of shape (1024,)

    Returns:
    torch.Tensor: The aggregated embedding of shape (1024,)
    """
    # Stack embeddings
    stacked_embeddings = torch.stack(embeddings)  # Shape: (num_embeddings, 1024)

    # Compute element-wise median
    aggregated_embedding = torch.median(stacked_embeddings, dim=0).values

    return aggregated_embedding

def percentile_aggregate_embeddings(embeddings, percentile=50):
    stacked_embeddings = torch.stack(embeddings)
    return torch.quantile(stacked_embeddings, q=percentile/100, dim=0).unsqueeze(0)

def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return np.asarray(pcd.points)


def calculate_chamfer_distances(folder_path):
    from tqdm.auto import tqdm
    from model_architectures.losses import cd_loss_l1
    gt_files = sorted([f for f in os.listdir(folder_path) if 'gt' in f])
    output_files = sorted([f for f in os.listdir(folder_path) if 'output' in f])

    if len(gt_files) != len(output_files):
        raise ValueError("Number of ground truth and output files do not match.")

    chamfer_distances = []

    for gt_file, output_file in tqdm(zip(gt_files, output_files), total=len(gt_files), desc="Processing"):
        gt_path = os.path.join(folder_path, gt_file)
        output_path = os.path.join(folder_path, output_file)

        gt_points = read_ply(gt_path)
        output_points = read_ply(output_path)

        gt_tensor = torch.from_numpy(gt_points).float().unsqueeze(0)
        output_tensor = torch.from_numpy(output_points).float().unsqueeze(0)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        gt_tensor = gt_tensor.to(device)
        output_tensor = output_tensor.to(device)
        print(f"Ground Truth shape: {gt_tensor.shape}")
        print(f"Output shape: {output_tensor.shape}")

        distance = cd_loss_l1(gt_tensor, output_tensor)  # Using your existing CD function
        chamfer_distances.append(distance)

    return chamfer_distances


def select_files_by_numbers(file_list, selected_numbers):
    import re
    """
    Select files from a list based on specific numbers in their filenames.

    Args:
    file_list (List[str]): List of filenames
    selected_numbers (List[int]): List of numbers to select files by

    Returns:
    List[str]: List of selected filenames
    """
    selected_files = []
    for file in file_list:
        # Extract numbers from the filename
        numbers = re.findall(r'\d+', file)
        if numbers:
            # Check if any of the extracted numbers are in the selected_numbers list
            if any(int(num) in selected_numbers for num in numbers):
                selected_files.append(file)
    return selected_files


def calc_cd_avg_gts(gt_files, output_pcd, folder_path):
    from model_architectures.losses import cd_loss_l1
    cds_average = []
    for gt_file in gt_files:
        gt_path = os.path.join(folder_path, gt_file)
        gt_points = read_ply(gt_path)
        gt_tensor = torch.from_numpy(gt_points).float().unsqueeze(0)
        output_tensor = torch.from_numpy(np.asarray(output_pcd.points)).float().unsqueeze(0)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        gt_tensor = gt_tensor.to(device)
        output_tensor = output_tensor.to(device)
        distance = cd_loss_l1(gt_tensor, output_tensor)
        cds_average.append(distance)
    return cds_average


import torch
import heapq
from typing import List


def find_smallest_10_tensors(tensor_list):
    """
    Find the 10 smallest values from a list of single-value tensors.
    Args:
    tensor_list (List[torch.Tensor]): List of tensors, each containing a single value
    Returns:
    List[tuple]: The 10 smallest values with their indices in the format (index, value)
    """
    # Convert tensors to (index, value) pairs
    indexed_values = [(i, tensor.item()) for i, tensor in enumerate(tensor_list)]
    # Use heapq to efficiently find the 10 smallest elements
    return heapq.nsmallest(10, indexed_values, key=lambda x: x[1])


def calculate_mean_of_tensors(tensor_list: List[torch.Tensor]) -> float:
    """
    Calculate the mean of a list of tensors, each containing a single value.

    Args:
    tensor_list (List[torch.Tensor]): List of tensors, each with a single value

    Returns:
    float: The mean value
    """
    # Convert each tensor to a scalar and create a new tensor
    values = torch.tensor([t.item() for t in tensor_list])

    # Calculate and return the mean
    return torch.mean(values).item()


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
