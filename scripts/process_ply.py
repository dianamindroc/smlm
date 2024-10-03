import os
import re
import pandas as pd
import numpy as np
import hdbscan
from plyfile import PlyData


def extract_sample_number(filename):
    match = re.search(r'sample(\d+)', filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


def find_matching_model_structure(model_folder, sample_number):
    pattern = f"model_structure{sample_number}"
    for file in os.listdir(model_folder):
        if pattern in file:
            return os.path.join(model_folder, file)
    return None


def read_model_coordinates(model_path):
    if model_path and os.path.exists(model_path):
        model_df = pd.read_csv(model_path)
        return model_df[['x', 'y', 'z']].iloc[0]  # Assuming we want the first row
    return pd.Series({'model_x': np.nan, 'model_y': np.nan, 'model_z': np.nan})


def process_ply_files(folder_path, model_folder, cluster_size=10, min_samples=5):
    all_centers_of_mass = []

    ply_files = [f for f in os.listdir(folder_path) if "output" in f.lower() and f.endswith(".ply")]

    for ply_file in ply_files:
        file_path = os.path.join(folder_path, ply_file)

        # Read PLY file
        plydata = PlyData.read(file_path)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']

        # Convert point cloud to DataFrame
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})

        # Perform clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=min_samples)
        df['cluster_label'] = clusterer.fit_predict(df[['x', 'y', 'z']])

        # Calculate centroids (centers of mass) for each cluster, excluding background (-1)

        centers_of_mass = df[df['cluster_label'] != -1].groupby('cluster_label').mean().reset_index()

        # If no valid clusters found, continue to next file
        if centers_of_mass.empty:
            print(f"No valid clusters found in {ply_file}. Skipping.")
            continue
        # Add file name to the centers_of_mass DataFrame
        centers_of_mass['file_name'] = ply_file

        # Extract sample number and find matching model structure
        sample_number = extract_sample_number(ply_file)
        if sample_number is not None:
            model_path = find_matching_model_structure(model_folder, sample_number)
            model_coords = read_model_coordinates(model_path)
            centers_of_mass['model_x'] = model_coords.get('x', np.nan)
            centers_of_mass['model_y'] = model_coords.get('y', np.nan)
            centers_of_mass['model_z'] = model_coords.get('z', np.nan)
        else:
            centers_of_mass['model_x'] = np.nan
            centers_of_mass['model_y'] = np.nan
            centers_of_mass['model_z'] = np.nan

        # Append to the list of all centers of mass
        all_centers_of_mass.append(centers_of_mass)

    return all_centers_of_mass


def extract_sample_number(filename):
    match = re.search(r'sample(\d+)', filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


def find_matching_model_structure(model_folder, sample_number):
    pattern = f"model_structure{sample_number}"
    for file in os.listdir(model_folder):
        if pattern in file:
            return os.path.join(model_folder, file)
    return None


def process_ply_files(folder_path, model_folder, cluster_size=10, min_samples=5):
    all_centers_of_mass = []

    ply_files = [f for f in os.listdir(folder_path) if "output" in f.lower() and f.endswith(".ply")]

    for ply_file in ply_files:
        file_path = os.path.join(folder_path, ply_file)

        # Read PLY file
        plydata = PlyData.read(file_path)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']

        # Convert point cloud to DataFrame
        df = pd.DataFrame({'x': x, 'y': y, 'z': z})

        # Perform clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=min_samples)
        df['cluster_label'] = clusterer.fit_predict(df[['x', 'y', 'z']])

        # Calculate centroids (centers of mass) for each cluster, excluding background (-1)

        centers_of_mass = df[df['cluster_label'] != -1].groupby('cluster_label').mean().reset_index()

        # If no valid clusters found, continue to next file
        if centers_of_mass.empty:
            print(f"No valid clusters found in {ply_file}. Skipping.")
            continue
        # Add file name to the centers_of_mass DataFrame
        centers_of_mass['file_name'] = ply_file

        # Extract sample number and find matching model structure
        sample_number = extract_sample_number(ply_file)
        if sample_number is not None:
            model_structure_path = find_matching_model_structure(model_folder, sample_number)
            centers_of_mass['model_structure_path'] = model_structure_path
        else:
            centers_of_mass['model_structure_path'] = None

        # Append to the list of all centers of mass
        all_centers_of_mass.append(centers_of_mass)

    return all_centers_of_mass