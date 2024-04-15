import open3d as o3d
import pandas as pd
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import hdbscan

import chart_studio.plotly as py
import chart_studio
chart_studio.tools.set_credentials_file(username='dianamindroc', api_key='7KZoUynf7MmNb5D5Jcf2')

pio.renderers.default = "browser"
pio.templates["custom_dark"] = go.layout.Template(
    # LAYOUT
    layout = {
        # Fonts
        'title': {'font': {'family': 'HelveticaNeue-CondensedBold, Helvetica, Sans-serif',
                           'size': 30,
                           'color': '#FFFFFF'}},  # White font color
        'font': {'family': 'Helvetica Neue, Helvetica, Sans-serif',
                 'size': 16,
                 'color': '#FFFFFF'},  # White font color
        # Background colors
        'paper_bgcolor': '#000000',  # Black paper background color
        'plot_bgcolor': '#000000',   # Black plot background color
        # Colorways
        'colorway': ['#ec7424', '#a4abab'],  # Custom colorway
        # Hover mode
        'hovermode': 'closest'
    },
    # DATA
    data = {
        # Each graph object must be in a tuple or list for each trace
        'scatter3d': [go.Scatter3d(marker=dict(color='gray'))]  # Set scatter3d marker color to gray
        # You can add more trace types here as needed
    }
)

# Set the default template to the custom one
pio.templates.default = 'custom_dark'
# Define the folder containing your .ply files
folder_path = '/home/dim26fa/coding/training/logs_pcn_20240409_111553/'

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
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=2)
        cluster_labels_pred = clusterer.fit_predict(pred_dfs[identifier][['x', 'y', 'z']])
        pred_dfs[identifier]['cluster_label'] = cluster_labels_pred
    elif 'gt' in file_name:
        identifier = file_name.split('_gt')[0]
        file_path = os.path.join(folder_path, file_name)
        point_cloud = o3d.io.read_point_cloud(file_path)
        gt_dfs[identifier] = pd.DataFrame(point_cloud.points, columns=['x', 'y', 'z']).drop_duplicates()
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=2)
        cluster_labels_gt = clusterer.fit_predict(gt_dfs[identifier][['x', 'y', 'z']])
        gt_dfs[identifier]['cluster_label'] = cluster_labels_gt

# Initialize the list to hold the pairs
datasets = []

# Assuming the keys in `pred_dfs` and `gt_dfs` match and represent the same samples
for identifier in pred_dfs.keys():
    # Check if the identifier exists in both dictionaries before pairing
    if identifier in gt_dfs:
        pred_df = pred_dfs[identifier]
        gt_df = gt_dfs[identifier]
        datasets.append((pred_df, gt_df))

# Setup for a 2x5 grid of 3D scatter plots
fig = make_subplots(rows=1, cols=3,
                    specs=[[{'type': 'scatter3d'} for _ in range(3)] for _ in range(1)],
                    horizontal_spacing=0.02, vertical_spacing=0.1)

for i, (pred, gt) in enumerate(datasets[6:9]):
    row = (i // 3) + 1  # Determine row for subplot
    col = (i % 3) + 1  # Determine column for subplot

    fig.add_trace(
        go.Scatter3d(
            x=pred['x'],
            y=pred['y'],
            z=pred['z'],
            mode='markers',
            marker=dict(size=2, color='#7030A0', opacity=0.5),
            name=f'predicted_{i}'
        ),
        row=row, col=col
    )
    centers_of_mass_pred = pred.groupby('cluster_label').mean().reset_index()

    fig.add_trace(
        go.Scatter3d(
            x=centers_of_mass_pred['x'],
            y=centers_of_mass_pred['y'],
            z=centers_of_mass_pred['z'],
            mode='markers+text',
            marker=dict(size=7, color='yellow', symbol='diamond', opacity=0.7),
            text=centers_of_mass_pred['cluster_label'],
            textposition='top center',
            name=f'Center of Mass_pred_{i}'
        ),
        row=row, col=col
    )
    ##################################
    fig.add_trace(
        go.Scatter3d(
            x=gt['x'],
            y=gt['y'],
            z=gt['z'],
            mode='markers',
            marker=dict(size=2, color='red', opacity=0.5),
            name=f'ground truth_{i}'
        ),
        row=row, col=col
    )
    centers_of_mass_gt = gt.groupby('cluster_label').mean().reset_index()

    fig.add_trace(
        go.Scatter3d(
            x=centers_of_mass_gt['x'],
            y=centers_of_mass_gt['y'],
            z=centers_of_mass_gt['z'],
            mode='markers+text',
            marker=dict(size=7, color='green', symbol='diamond', opacity=0.7),
            text=centers_of_mass_gt['cluster_label'],
            textposition='top center',
            name=f'Center of Mass_gt_{i}'
        ),
        row=row, col=col
    )
# Show the figure
fig.show()


py.plot(fig, filename = 'test1', auto_open=True)