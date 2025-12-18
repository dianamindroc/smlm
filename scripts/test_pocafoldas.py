import argparse
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
from joblib import dump, load

from helpers.data import get_highest_shape
from helpers.logging import create_log_folder
from model_architectures.losses import l1_cd, l2_cd, f_score
from model_architectures.pcn_decoder import PCNDecoderOnly
from model_architectures.pocafoldas import PocaFoldAS
from model_architectures.transforms import Padding, ToTensor
from model_architectures.utils import cfg_from_yaml_file, l1_cd_metric
from dataset.Dataset import PairedAnisoIsoDataset
from dataset.ShapeNet import ShapeNet
from dataset.SMLMDataset import Dataset

# Legacy exploratory code that depended on local files and undefined variables.
# Disabled to keep the script importable and testable.
if False:  # pragma: no cover
    ckpt = torch.load('/home/dim26fa/coding/training/logs_pcn_20250326_161041/best_l1_cd.pth')
    model = PocaFoldAS(classifier=True)
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
    tensor_vectors = [torch.tensor(elem) for elem in feat_space]
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
    model = PocaFoldAS(num_dense=16384, latent_dim=1024, grid_size=4, classifier=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def setup_transformations(highest_shape):
    """ Set up data transformations. """
    return transforms.Compose([
        Padding(highest_shape), 
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
            inputs = inputs.permute(0, 2, 1)  
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
    """
    Test a model on a single category of point cloud data.

    Args:
        model: The PocaFoldAS model to test
        device: Device to run inference on (cuda/cpu)
        device: Device to run inference on (cuda/cpu)
        test_config: Configuration settings for testing
        log_dir: Directory to save results
        save: Whether to save output point clouds as PLY files

    Returns:
        Dictionary containing metrics and result arrays
    """
    # Get shape information and prepare transforms
    highest_shape = get_highest_shape(test_config['root_folder'], test_config['classes_to_use'])
    pc_transforms = transforms.Compose([Padding(highest_shape), ToTensor()])

    # Create dataset and dataloader
    test_dataset = PairedAnisoIsoDataset(
        root_folder=test_config['root_folder'],
        suffix=test_config['suffix'],
        transform=pc_transforms,
        classes_to_use=test_config['classes_to_use'],
        remove_part_prob=test_config['remove_part_prob'],
        remove_corners=test_config['remove_corners'],
        remove_outliers=test_config['remove_outliers'],
        number_corners_to_remove=test_config['num_corners_remove']
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # Initialize metrics and result containers
    total_l1_cd, total_l2_cd = 0.0, 0.0
    feature_space = []
    labels_list = []
    corner_label_list = []
    corner_name = []
    l1cd_list = []
    l2cd_list = []
    out_classifier_list = []
    labels_names = []

    # Process each sample
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            # Extract and prepare input data
            input_pc = data['partial_pc'][:, :, :3].permute(0, 2, 1).to(device)
            gt_pc = data['pc'][:, :, :3].to(device)
            filename = os.path.basename(data['filename'][0])
            #label = data['label'].to(device)
            #corner_label = data['corner_label'].to(device)

            # Forward pass through model
            _, output_pc, feature_global, out_classifier, attn_weights = model(input_pc)

            # Calculate metrics
            l1_error = l1_cd(output_pc, gt_pc).item()
            l2_error = l2_cd(output_pc, gt_pc).item()
            total_l1_cd += l1_error
            total_l2_cd += l2_error
            l1cd_list.append(l1_error)
            l2cd_list.append(l2_error)

            # Store results
            labels_list.append(data['label'].numpy())
            labels_names.append(data['label_name'])
            corner_label_list.append(data['corner_label'].numpy())
            corner_name.append(data['corner_label_name'])
            feature_space.append(feature_global[0].detach().cpu().numpy())
            out_classifier_list.append(F.softmax(out_classifier, dim=1).detach().cpu().numpy())

            # Save output point clouds if requested
            if save:
                # Detach and move tensors to CPU for saving
                input_np = input_pc[0].permute(1, 0).detach().cpu().numpy()
                output_np = output_pc[0].detach().cpu().numpy()
                gt_np = gt_pc[0].detach().cpu().numpy()
                attn_matrix = attn_weights.detach().cpu().numpy()

                # Save as PLY files
                export_ply(os.path.join(log_dir, f'{i + 1:03d}_input_{filename}.ply'), input_np)
                export_ply(os.path.join(log_dir, f'{i + 1:03d}_output{filename}.ply'), output_np)
                export_ply(os.path.join(log_dir, f'{i + 1:03d}_gt{filename}.ply'), gt_np)
                np.save(os.path.join(log_dir, f'{i + 1:03d}_attn_matrix.npy'), attn_matrix)

    # Calculate average metrics
    dataset_size = len(test_dataset)
    avg_l1_cd = total_l1_cd / dataset_size
    avg_l2_cd = total_l2_cd / dataset_size

    # Consolidate results
    labels_array = np.concatenate(labels_list, axis=0)
    labels_names = np.concatenate(labels_names)
    corner_array = np.concatenate(corner_label_list)
    corner_name = np.concatenate(corner_name)
    out_classifier_array = np.concatenate(out_classifier_list, axis=0)
    predicted_class_array = np.argmax(out_classifier_array, axis=1)

    # Return results as a dictionary
    return {
        'avg_l1_cd': avg_l1_cd,
        'avg_l2_cd': avg_l2_cd,
        'feature_space': feature_space,
        'labels': labels_array,
        'label_names': labels_names,
        'corner_labels': corner_array,
        'corner_names': corner_name,
        'classifier_outputs': out_classifier_array,
        'predicted_classes': predicted_class_array,
        'l1_errors': np.array(l1cd_list),  # Including individual errors for analysis
        'l2_errors': np.array(l2cd_list)
    }


def test(config, save=True):
    """
    Run testing on the point cloud model using the provided configuration.

    Args:
        config: Path to YAML config file or config object
        save: Whether to save test results (default: True)

    Returns:
        Dictionary containing test metrics and result data
    """
    # Load configuration if string path provided
    if isinstance(config, str):
        config = cfg_from_yaml_file(config)

    # Extract test configuration for the dataset
    test_config = {
        'root_folder': config.dataset.root_folder,
        'classes_to_use': config.dataset.classes,
        'save': config.test.save,
        'ckpt_path': config.test.ckpt_path,
        'ckpt_path_class': config.test.ckpt_path_class,
        'suffix': config.dataset.suffix,
        'remove_part_prob': config.dataset.remove_part_prob,
        'remove_corners': config.dataset.remove_corners,
        'anisotropy': config.dataset.anisotropy,
        'anisotropy_factor': config.dataset.anisotropy_factor,
        'anisotropy_axis': config.dataset.anisotropy_axis,
        'remove_outliers': config.dataset.remove_outliers,
        'classifier': config.test.classifier,
        'num_corners_remove': config.dataset.number_corners_remove,
        'grid_size': config.test.grid_size,
        'num_dense': config.test.num_dense,
        'latent_dim': config.test.latent_dim,
        'channels': config.test.channels
    }

    # Create log directory if saving results
    log_dir = create_log_folder(config.test.log_dir, config.model) if save else None

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if test_config['num_dense'] == 'highest':
        num_dense = get_highest_shape(test_config['root_folder'], test_config['classes_to_use'])
        print(f'Highest shape: {num_dense}')
    else:
        num_dense = test_config['num_dense']

    # Print header for metrics table
    print('\033[33m{:20s}{:20s}{:20s}\033[0m'.format('Category', 'L1_CD(1e-3)', 'L2_CD(1e-4)'))
    print('\033[33m{:20s}{:20s}{:20s}\033[0m'.format('--------', '-----------', '-----------'))

    # Initialize model with configurable parameters
    model_recon = PocaFoldAS(num_dense, test_config['latent_dim'], test_config['grid_size'], classifier=config.test.classifier).to(device)
    recon_state_dict = torch.load(config.test.ckpt_path)

    # Handle loading models with incompatible keys
    if 'mlp3.weight' in recon_state_dict.keys() and config.test.classifier is False:
        keys_to_remove = [
            "mlp1.weight", "mlp1.bias",
            "bn1_mlp.weight", "bn1_mlp.bias", "bn1_mlp.running_mean", "bn1_mlp.running_var",
            "bn1_mlp.num_batches_tracked",
            "mlp2.weight", "mlp2.bias",
            "bn2_mlp.weight", "bn2_mlp.bias", "bn2_mlp.running_mean", "bn2_mlp.running_var",
            "bn2_mlp.num_batches_tracked",
            "mlp3.weight", "mlp3.bias"
        ]

        for key in keys_to_remove:
            if key in recon_state_dict:
                del recon_state_dict[key]

    expected_dim = model_recon.mlp[4].bias.shape[0]
    ckpt_dim = recon_state_dict['mlp.4.bias'].shape[0]
    print(f'Expected dimension: {expected_dim}, ckpt dimension: {ckpt_dim}')

    #if ckpt_dim != expected_dim:
        #num_dense = ckpt_dim
        #model_recon = PocaFoldAS(num_dense, test_config['latent_dim'], test_config['grid_size'],
                          #classifier=config.test.classifier).to(device)

    model_recon.load_state_dict(recon_state_dict, strict=False)
    model_recon.eval()

    results_recon = test_single_category(model_recon, device, test_config, log_dir, save)
    print('\033[32m{:20s}{:20.4f}{:20.4f}\033[0m'.format(
        'Recon Model', results_recon['avg_l1_cd'] * 1e3, results_recon['avg_l2_cd'] * 1e4
    ))
    # Load checkpoint
    model_class = PocaFoldAS(num_dense, test_config['latent_dim'], test_config['grid_size'],
                      classifier=config.test.classifier).to(device)
    class_state_dict = torch.load(config.test.ckpt_path_class)

    model_class.load_state_dict(class_state_dict)
    model_class.eval()

    results_class = test_single_category(model_class, device, test_config, log_dir,
                                         save=False)  # Don't save files twice
    print('\033[32m{:20s}{:20.4f}{:20.4f}\033[0m'.format(
        'Classifier Model', results_class['avg_l1_cd'] * 1e3, results_class['avg_l2_cd'] * 1e4
    ))





    # Run test on the dataset
    #results = test_single_category(model, device, test_config, log_dir, save)

    # Save results dictionary if log directory exists
    if log_dir and save:
        import pickle
        import json
        # Save as pickle (preserves numpy arrays and python objects)
        with open(os.path.join(log_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results_recon, f)

        # Additionally save metrics as JSON for easy viewing
        # Need to convert numpy arrays to lists for JSON serialization
        json_safe_results = {}
        for key, value in results_recon.items():
            if isinstance(value, np.ndarray):
                # Handle different array shapes
                if value.ndim == 1:
                    json_safe_results[key] = value.tolist()
                else:
                    # For higher dimensional arrays, just save shape information
                    json_safe_results[key] = f"Array shape: {value.shape}"
            elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                # For lists of arrays (like feature_space)
                json_safe_results[key] = f"List of {len(value)} arrays, first shape: {value[0].shape}"
            elif isinstance(value, (int, float, str, bool, list)):
                json_safe_results[key] = value
            else:
                json_safe_results[key] = str(value)

        # Save the JSON-compatible subset of results
        with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
            json.dump(json_safe_results, f, indent=2)

        print(f"Results saved to {log_dir}")

    # Print overall metrics with green highlighting
    #print('\033[32m{:20s}{:20.4f}{:20.4f}\033[0m'.format(
    #    'All', results['avg_l1_cd'] * 1e3, results['avg_l2_cd'] * 1e4
    #))

    return {
        'reconstruction_model': results_recon,
        'classifier_model': results_class
    }


import torch.nn as nn

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


def get_min_chamfer(l1_errors):
    min_l1 = float('inf')  # Initialize with infinity to find minimum
    index_min = -1  # Initialize with invalid index

    for i, l1 in enumerate(l1_errors):
        if i < len(l1_errors) - 1:  # Check to prevent index out of range
            if l1_errors[i + 1] < l1:  # Compare next value to current
                if l1_errors[i + 1] < min_l1:  # Update only if it's the new minimum
                    min_l1 = l1_errors[i + 1]
                    index_min = i + 1

    return {
        'min_chamfer': min_l1,
        'index_min': index_min
            }



def _parse_args():
    parser = argparse.ArgumentParser('Point Cloud Completion Testing')
    parser.add_argument('--config', type=str, default='configs/test_config.yaml', help='Path to YAML config')
    parser.add_argument('--save', action='store_true', help='Save PLY/metrics outputs')
    return parser.parse_args()


def _main():
    args = _parse_args()
    results = test(args.config, save=args.save)
    print('Finished testing. Metrics keys:', list(results.keys()))


if __name__ == '__main__':
    _main()
