# script with quick code snippets for testing purposes
from plotly.subplots import make_subplots

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

pc1 = o3d.io.read_point_cloud('/home/dim26fa/coding/training/logs_pcn_20240530_163253/100_gt.ply')
pc2 = o3d.io.read_point_cloud('/home/dim26fa/coding/training/logs_pcn_20240530_163253/100_output.ply')
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

tree = KDTree(arr2)
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

### Reading gt + pred
import open3d as o3d
import pandas as pd
import numpy as np
import os
pred_ply = o3d.io.read_point_cloud('/home/dim26fa/coding/testing/logs_pcn_20240420_134549/104_output.ply')
gt_ply = o3d.io.read_point_cloud('/home/dim26fa/coding/testing/logs_pcn_20240420_134549/104_gt.ply')
pred = pd.DataFrame(pred_ply.points, columns=['x', 'y', 'z'])
gt = pd.DataFrame(gt_ply.points, columns=['x', 'y', 'z'])
pred = pred.drop_duplicates()
pred = np.array(pred)
gt = np.array(gt_ply.points)

# Define the folder containing your .ply files
folder_path = '/home/dim26fa/coding/testing/logs_pcn_20240420_134549/'

# List all .ply files in the folder
ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
pred_dfs = {}
gt_dfs = {}

# Process files and convert to pandas DataFrame
for file_name in ply_files:
    # Separate into pred and gt based on file naming convention
    if 'output' in file_name:
        identifier = file_name.split('_output')[0]  # This assumes a specific part of the file name can serve as an identifier
        file_path = os.path.join(folder_path, file_name)
        point_cloud = o3d.io.read_point_cloud(file_path)
        pred_dfs[identifier] = pd.DataFrame(point_cloud.points, columns=['x', 'y', 'z']).drop_duplicates()
    elif 'gt' in file_name:
        identifier = file_name.split('_gt')[0]
        file_path = os.path.join(folder_path, file_name)
        point_cloud = o3d.io.read_point_cloud(file_path)
        gt_dfs[identifier] = pd.DataFrame(point_cloud.points, columns=['x', 'y', 'z']).drop_duplicates()

# Initialize the list to hold the pairs
datasets = []

# Assuming the keys in `pred_dfs` and `gt_dfs` match and represent the same samples
for identifier in pred_dfs.keys():
    # Check if the identifier exists in both dictionaries before pairing
    if identifier in gt_dfs:
        pred_df = pred_dfs[identifier]
        gt_df = gt_dfs[identifier]
        datasets.append((pred_df, gt_df))

for i, (pred, new_data) in enumerate(datasets, start=1):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=pred['x'],
            y=pred['y'],
            z=pred['z'],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.5),
            name=f'Dataset 1 - Plot {i}'
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=new_data['x'],
            y=new_data['y'],
            z=new_data['z'],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.5),
            name=f'Dataset 2 - Plot {i}'
        )
    )

    fig.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(centers_of_mass_pred['x'], centers_of_mass_pred['y'], centers_of_mass_pred['z'], color='red', marker='o', label='pred')
ax.scatter(centers_of_mass_gt['x'], centers_of_mass_gt['y'], centers_of_mass_gt['z'], color='blue', marker='o', label='gt')
plt.legend()
plt.savefig('/home/dim26fa/data/to_plot/centers.png')
plt.show()

import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
import plotly.graph_objects as go
fig = px.scatter_3d(pred, x='x', y='y', z='z', size_max=1, color='red')
fig.add_trace(px.scatter_3d(gt, x='x', y='y', z='z', size_max=1).data[0], color='blue')
fig.show()

# Your first dataset plotted with Plotly Express for convenience
# Creating the figure object
fig = go.Figure()

fig = make_subplots(rows=2, cols=5,
                    specs=[[{'type': 'scatter3d'} for _ in range(5)] for _ in range(2)],
                    subplot_titles=[f'Pair {i+1}' for i in range(10)],
                    horizontal_spacing=0.02, vertical_spacing=0.1)

# Adding the first dataset with a specified name and color
fig.add_trace(
    go.Scatter3d(
        x=pred['x'],
        y=pred['y'],
        z=pred['z'],
        mode='markers',
        marker=dict(
            size=3,  # Adjust the size as needed
            color='blue',  # Specifying color for the first dataset
            opacity=0.5
        ),
        name='predicted'  # Naming the first dataset
    )
)

# Adding the second dataset with a different specified name and color
fig.add_trace(
    go.Scatter3d(
        x=gt['x'],
        y=gt['y'],
        z=gt['z'],
        mode='markers',
        marker=dict(
            size=3,  # Adjust the size as needed
            color='red',  # Specifying color for the second dataset
            opacity=0.5
        ),
        name='ground truth'  # Naming the second dataset
    )
)

clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=2)
cluster_labels_gt = clusterer.fit_predict(gt[['x', 'y', 'z']])
gt['cluster_label'] = cluster_labels_gt
centers_of_mass_gt = gt.groupby('cluster_label').mean().reset_index()
fig.add_trace(
    go.Scatter3d(
        x=centers_of_mass_gt['x'],
        y=centers_of_mass_gt['y'],
        z=centers_of_mass_gt['z'],
        mode='markers+text',
        marker=dict(size=7, color='black', symbol='diamond', opacity=0.7),
        text=centers_of_mass_gt['cluster_label'],
        textposition='top center',
        name='Center of Mass - GT'
    )
)

clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=2)
cluster_labels_pred = clusterer.fit_predict(pred[['x', 'y', 'z']])
pred['cluster_label'] = cluster_labels_pred
centers_of_mass_pred = pred.groupby('cluster_label').mean().reset_index()
fig.add_trace(
    go.Scatter3d(
        x=centers_of_mass_pred['x'],
        y=centers_of_mass_pred['y'],
        z=centers_of_mass_pred['z'],
        mode='markers+text',
        marker=dict(size=7, color='green', symbol='diamond', opacity=0.7),
        text=centers_of_mass_pred['cluster_label'],
        textposition='top center',
        name='Center of Mass - PRED'
    )
)
# Display the figure
fig.show()

#### Viz stats

# Ensure your stats_list is available with all the necessary statistics

# Histogram of the Number of Points using Seaborn
if style=='dark':
    sns.set_style("dark", {'axes.facecolor': 'black', 'figure.facecolor': 'black'})
    clr = 'white'
else:
    sns.set_style("white")
    clr = 'black'
num_points = [stats['num_points'] for stats in stats]
sns.histplot(num_points, bins=20, kde=True, color='steelblue')  # KDE (Kernel Density Estimate) adds a smoothed line to help understand the distribution shape
plt.title('Distribution of Number of Points - exp_tetra', color=clr)
plt.xlabel('Number of Points', color=clr)
plt.ylabel('Frequency', color=clr)
if style=='dark':
    plt.tick_params(colors=clr)
plt.show()

# Box Plot for Bounding Box Sizes with Seaborn
# Convert bounding box sizes to a DataFrame for easier plotting with seaborn
bounding_box_sizes_df = pd.DataFrame([stats['bounding_box_size'] for stats in stats], columns=['x', 'y', 'z'])
# Melt the DataFrame to long-format for seaborn's boxplot
bounding_box_sizes_melted = bounding_box_sizes_df.melt(var_name='Dimension', value_name='Size')

sns.boxplot(x='Dimension', y='Size', data=bounding_box_sizes_melted,
            color='steelblue',  # Color of the box fill
            linewidth=1,    # Thickness of the lines
            fliersize=5,     # Size of the outlier markers
            flierprops={'markerfacecolor':'white', 'markeredgecolor':'white'},  # Outlier styles
            medianprops={'color':'white'},  # Median line style
            whiskerprops={'color':'white'},  # Whisker line style
            capprops={'color':'white'},      # Cap line style
            boxprops={'edgecolor':'white'})  # Box edge line style)
plt.title('Bounding Box Size Distribution - exp_tetra', color=clr)
if style=='dark':
    plt.tick_params(colors=clr)
plt.xlabel('Dimension', color=clr)
plt.ylabel('Size', color=clr)
plt.show()

std_dev_df = pd.DataFrame([stats['std_dev'] for stats in stats], columns=['x', 'y', 'z'])

std_dev_melted = std_dev_df.melt(var_name='Dimension', value_name='Standard Deviation')

sns.boxplot(x='Dimension', y='Standard Deviation', data=std_dev_melted,
            color='steelblue',
            linewidth=1, flierprops={'markerfacecolor':'white', 'markeredgecolor':'white'},
            medianprops={'color':'white'}, whiskerprops={'color':'white'},
            capprops={'color':'white'}, boxprops={'edgecolor':'white'})

plt.title('Standard Deviation Distribution by Dimension - exp_tetra', color='white')
plt.xlabel('Dimension', color='white')
plt.ylabel('Standard Deviation', color='white')
plt.tick_params(colors='white')
plt.show()

eigenvalues_df = pd.DataFrame([stats['eigenvalues'] for stats in stats], columns=['1st Principal', '2nd Principal', '3rd Principal'])
eigenvalues_melted = eigenvalues_df.melt(var_name='Principal Component', value_name='Eigenvalue')

sns.boxplot(x='Principal Component', y='Eigenvalue', data=eigenvalues_melted,
            color='steelblue',
            linewidth=1, flierprops={'markerfacecolor':'white', 'markeredgecolor':'white'},
            medianprops={'color':'white'}, whiskerprops={'color':'white'},
            capprops={'color':'white'}, boxprops={'edgecolor':'white'})

plt.title('Eigenvalue Distribution by Principal Component - exp_tetra', color='white')
plt.xlabel('Principal Component', color='white')
plt.ylabel('Eigenvalue', color='white')
plt.tick_params(colors='white')
plt.show()

df = pd.DataFrame(pd.read_csv(pcs[1]))

# Setting the background style for better visibility
if style=='dark':
    sns.set_style("dark", {'axes.facecolor': 'black', 'figure.facecolor': 'black'})
    clr = 'white'
else:
    sns.set_style("white")
    clr = 'black'

# Plotting histograms
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['x'], kde=True, color='r')
plt.title('Distribution of Points in X Direction', color=clr)
plt.xlabel('x', color=clr)
plt.ylabel('Count', color=clr)
plt.tick_params(colors=clr)

plt.subplot(1, 3, 2)
sns.histplot(df['y'], kde=True, color='g')
plt.title('Distribution of Points in Y Direction', color=clr)
plt.xlabel('y', color=clr)
plt.ylabel('Count', color=clr)
plt.tick_params(colors=clr)

plt.subplot(1, 3, 3)
sns.histplot(df['z'], kde=True, color='b')
plt.title('Distribution of Points in Z Direction', color=clr)
plt.xlabel('z', color=clr)
plt.ylabel('Count', color=clr)
plt.tick_params(colors=clr)

plt.tight_layout()
plt.show()

# Plotting box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title('Box Plot of Point Distribution in X, Y, Z Directions')
plt.ylabel('Value')
plt.show()


import pandas as pd
import numpy as np

# Assuming you have a list of point cloud arrays
point_clouds = glob.glob('/home/dim26fa/data/simulated_dna_origami/pyramid/*') # Example list of 5 point clouds

# Convert each point cloud to a DataFrame and add an identifier
dfs = []
for idx, pc in enumerate(point_clouds, start=1):
    df = pd.read_csv(pc)
    df['Point Cloud'] = f'Cloud {idx}'
    dfs.append(df)

# Combine into a single DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

sns.set(style="darkgrid", palette="colorblind")

# Plotting histograms for X, Y, Z directions for all point clouds
plt.figure(figsize=(18, 5))

for i, direction in enumerate(['x', 'y', 'z'], start=1):
    plt.subplot(1, 3, i)
    sns.histplot(data=combined_df, x=direction, hue='Point Cloud', kde=True, element='step', palette='colorblind', legend=False)
    plt.title(f'Distribution of Points in {direction} Direction')

plt.show()

import os
import pandas as pd

appends = ['5nm_cube', '10nm_cube', '20nm_cube', '5nm_pyramid', '10nm_pyramid', '20nm_pyramid']
for app in appends:
    path = '/home/dim26fa/data/dna_origami_one_sample512/test/partial/all'
#path = '/home/dim26fa/data/sim_dna_multiple_sizes_shapenet_version/test/partial/all'
    pcs = [os.path.join(path, entry) for entry in os.listdir(path) if entry.endswith('.csv')]
    pcs_array = [pd.read_csv(pc) for pc in pcs]
    partial_pcs = [_remove_corners(pc) for pc in pcs_array]
    for i, pc in enumerate(partial_pcs):
        pathh = '/home/dim26fa/data/dna_origami_one_sample512/test/partial/all/smlm_sample'
        pc.to_csv(pathh + str(i) + '.csv', index=False)

import h5py
appends = ['0', '1', '2', '3', '4', '5']
for app in appends:
    path = '/home/dim26fa/data/sim_dna_multiple_sizes_shapenet_version/val/gt/' + app
    path = '/home/dim26fa/data/dna_origami_one_sample512/test/partial/all'
    pcs = [os.path.join(path, entry) for entry in os.listdir(path) if entry.endswith('.csv')]
    for i, pc in enumerate(pcs):
        df = pd.read_csv(pc)
        structured_array = np.array(df)
        num_points_needed = 2048- structured_array.shape[0]
        if num_points_needed > 0:
            # Select random indices from the point cloud to duplicate
            indices_to_duplicate = np.random.choice(structured_array.shape[0], num_points_needed, replace=True)
            duplicated_points = structured_array[indices_to_duplicate]

            # Concatenate the original point cloud with the duplicated points
            padded_point_cloud = np.vstack((structured_array, duplicated_points))
        else:
            padded_point_cloud = structured_array

        # Create a mask indicating original points with ones and duplicated points with zeros

        pathh = '/home/dim26fa/data/dna_origami_one_sample2048/test/partial/all/'
        #df.to_hdf(pathh + str(i) + '_hdf.h5', key='data')
        with h5py.File(pathh + str(i) + '_hdf.h5', 'w') as file:
            # Create a dataset directly under the root of the HDF5 file
            file.create_dataset('data', data=padded_point_cloud)

folder_path = '/home/dim26fa/data/dna_origami_one_sample512/train/gt'
# Specify the path for the output text file
output_file_path = '/home/dim26fa/data/dna_origami_one_sample512/train/filenames.txt'

parent_folder_name = os.path.basename(folder_path)

with open(output_file_path, 'w') as file:
    # List all files in the directory
    for filename in os.listdir(folder_path):
        # Check if the file has the .h5 extension
        if filename.endswith('.h5'):
            # Remove the file extension
            filename_without_suffix, _ = os.path.splitext(filename)
            # Append parent folder name to filename
            full_filename = f"{parent_folder_name}/{filename_without_suffix}"
            # Write each full filename to the text file, followed by a newline
            file.write(full_filename + '\n')

with open(output_file_path, 'w') as file:
    # Iterate over each directory within the parent folder
    for folder_name in os.listdir(folder_path):
        # Construct the path to the subfolder
        _path = os.path.join(folder_path, folder_name)
        if os.path.isdir(_path):  # Check if it's a directory
            # List all files in the subfolder
            for filename in os.listdir(_path):
                # Check if the file has the .h5 extension
                if filename.endswith('.h5'):
                    # Remove the file extension
                    filename_without_suffix, _ = os.path.splitext(filename)
                    # Create the entry to write, including subfolder name and filename without suffix
                    entry = f"{folder_name}/{filename_without_suffix}"
                    # Write each entry to the text file, followed by a newline
                    file.write(entry + '\n')

root_dir = '/home/dim26fa/data/dna_origami_one_sample512/test/partial/all'

# Initialize the maximum length variable
max_length = 0

# Loop through each sub-folder in the root directory
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):  # Check if it's a directory
        # Loop through each file in the sub-folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):  # Make sure to process only CSV files
                file_path = os.path.join(folder_path, file_name)
                # Read the CSV file
                df = pd.read_hdf(file_path)
                # Update the maximum length if the current one is larger
                if len(df) > max_length:
                    max_length = len(df)

# Print the result
print(f"The maximum number of rows in any CSV file is: {max_length}")



### Scatter plot 3D with matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Sample data

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(df['x'], df['y'], df['z'])

# Labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

plt.show()


def plot_html(df):
    import plotly.graph_objects as go

    # Sample data
    x = df['x']
    y = df['y']
    z = df['z']

    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=z,                # Set color to z coordinate values for gradient effect
            colorscale='Viridis',   # Choose a colorscale
            opacity=0.8
        )
    )])

    # Update layout for a better look
    fig.update_layout(
        title='Interactive 3D Scatter Plot',
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        autosize=False,
        width=700,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90)
    )

    fig.show()
    fig.write_html('/home/dim26fa/coding/smlm/plot.html')


#TODO: check npcs from heydarian - how they look
#TODO: try out heydarian script with dna origami

