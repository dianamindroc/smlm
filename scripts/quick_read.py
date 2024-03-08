# script with quick code snippets for testing purposes

from helpers.readers import ReaderIMODfiles
import os
from helpers.visualization import print_pc_from_filearray

reader = ReaderIMODfiles('/home/dim26fa/data/imod_shapes/21022024/rectangle/')

from helpers.visualization import print_pc_from_filearray

path = reader.path_folder
path = '/home/dim26fa/data/imod_shapes/square'
pcs = [os.path.join(path, entry) for entry in os.listdir(path) if entry.endswith('.csv')]

print_pc_from_filearray(pcs[1:5])



import open3d as o3d

pc1 = o3d.io.read_point_cloud('/home/dim26fa/coding/training/logs_pcn_20240222_144847/010_output.ply')
pc1 = o3d.io.read_point_cloud('/home/dim26fa/coding/testing/airplane/output/001.ply')
pc2 = o3d.io.read_point_cloud('/home/dim26fa/coding/testing/new_weights/airplane/output/002.ply')
o3d.visualization.draw_geometries([pc1])


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

import numpy as np
from scipy.spatial.distance import cdist

def remove_statistical_outliers(points, k=50, std_multiplier=1.0):
    """
    Remove statistical outliers from a point cloud.

    Args:
        points (numpy.ndarray): The input point cloud as an Nx3 numpy array.
        k (int): Number of nearest neighbors to consider for each point.
        std_multiplier (float): The threshold in standard deviations to classify as an outlier.

    Returns:
        numpy.ndarray: The filtered point cloud with outliers removed.
    """
    # Compute pairwise distances
    distances = cdist(points, points, metric='euclidean')

    # Sort distances for each point and take the k smallest distances (excluding the point itself)
    sorted_distances = np.sort(distances, axis=1)[:, 1:k+1]

    # Compute the mean distance to k nearest neighbors for each point
    mean_distances = np.mean(sorted_distances, axis=1)

    # Compute the threshold for being an outlier
    threshold = np.mean(mean_distances) + std_multiplier * np.std(mean_distances)

    # Filter points that are below the threshold
    filtered_points = points[mean_distances < threshold]

    return filtered_points

# extract non-zero rows from array
def non_zero_rows(array):
    return array[~np.all(array == 0, axis=1)]

arr = np.array(pc1.points)

import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=None, algorithm='best', alpha=0.7,metric='euclidean')
clusterer.fit(arr)
non_outliers_indices = np.where(clusterer.labels_ != -1)[0]
nonoutlierarr = arr[non_outliers_indices]

cluster_labels = clusterer.fit_predict(arr)

# Create colormap for clusters
num_clusters = len(np.unique(cluster_labels))
colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))[:, :3]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(arr)
pcd.colors = o3d.utility.Vector3dVector(colors[cluster_labels])  # Assign colors based on cluster labels

# Visualize point cloud
o3d.visualization.draw_geometries([pcd])

from collections import Counter
label_counts = Counter(cluster_labels)

# Print the length of points in each label
for label, count in label_counts.items():
    print(f"Label {label}: {count} points")

selected_points = arr[cluster_labels == 1]

filtered_pc = arr[clusterer.outlier_scores_ < 0.8]
# Create Open3D point cloud from selected points
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_pc)
o3d.visualization.draw_geometries([pcd])

vis = o3d.visualization.Visualizer()

# Add the point cloud to the visualization
vis.create_window()
vis.add_geometry(pcd)

# Add point annotations with coordinates
for i, point in enumerate(pcd.points):
    label = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=point)
    vis.add_geometry(label)
# Visualize the selected cluster
vis.run()



root_folder = '/home/dim26fa/data/imod_shapes/21022024/'
suffix = '.csv'

classes = 'all'
highest_shape = 25734
pc_transforms = transforms.Compose(
    [Padding(highest_shape), ToTensor()])
full_dataset = Dataset(root_folder=root_folder, suffix=suffix, transform=pc_transforms, classes_to_use=classes, data_augmentation=True, remove_part_prob = 0.4, remove_outliers=True)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=6)






from helpers.visualization import Visualization3D
df = pd.DataFrame(arr)
vis = Visualization3D()


#### Try out chamfer distance
import torch
from model_architectures.chamfer_distance_updated import ChamferDistance
import open3d as o3d
import numpy as np
import pandas as pd

pc1 = o3d.io.read_point_cloud('/home/dim26fa/coding/training/logs_pcn_20240222_144847/010_input.ply')
pc2 = o3d.io.read_point_cloud('/home/dim26fa/coding/training/logs_pcn_20240222_144847/020_input.ply')
pc1 = pd.read_csv('/home/dim26fa/data/imod_shapes/21022024/rectangle/sample_17.csv')
pc2 = pd.read_csv('/home/dim26fa/data/imod_shapes/21022024/rectangle/sample_18.csv')
arr1 = np.array(pc1)
arr2 = np.array(pc2)
target_shape = (highest_shape, arr1.shape[-1])  # Assuming point cloud shape is (num_points, num_dimensions)
padding = [(0, target_shape[0] - arr1.shape[0]), (0, 0)]  # Calculate padding for points and dimensions
padded_arr = np.pad(arr2, padding, mode='constant', constant_values=0)
arr1 = torch.tensor(arr1, dtype=torch.float)
arr2 = torch.tensor(arr2, dtype=torch.float)
arr1 = arr1.unsqueeze(0)
arr2 = arr2.unsqueeze(0)
arr1 = arr1.to(device)
arr2 = arr2.to(device)
padded_arr_tensor = torch.tensor(padded_point_cloud, dtype=torch.float)
padded_arr_tensor = padded_arr_tensor.unsqueeze(0)
padded_arr_tensor = padded_arr_tensor.to(device)
dist1, dist2 = CD(arr1, arr2)
dist1 = torch.sqrt(dist1)
dist2 = torch.sqrt(dist2)
cdl1 = (torch.mean(dist1) + torch.mean(dist2)) / 2.0

## print overlay PCs
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], color='red', marker='o', label='pred')
ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], color='blue', marker='o', label='gt')
plt.show()

from scipy.spatial import KDTree

tree = KDTree(gt)
# Query the KD-tree for each source point to find its nearest neighbor in the target point cloud
distances, _ = tree.query(unique_pred)


# Normalize errors for color mapping
normalized_errors = distances / np.max(distances)

# Map normalized error values to a colormap
colors = plt.cm.jet(normalized_errors)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(unique_pred[:, 0], unique_pred[:, 1], unique_pred[:, 2], color=colors, marker='o')

cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])  # [left, bottom, width, height]
fig.subplots_adjust(right=0.8)  # Adjust main plot to make space for the colorbar

# Add a color bar to interpret the error magnitudes
mappable = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=np.min(distances), vmax=np.max(distances)))
mappable.set_array(distances)
cbar = fig.colorbar(mappable, cax=cbar_ax, aspect=5)
cbar.set_label('Error')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()