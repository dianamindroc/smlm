import plotly.express as px
import open3d as o3d
import pandas as pd
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
plt.ioff()


class Visualization3D:
    """
    Class for visualizing point cloud data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scatter = []

    def get_3d_scatter(self, size=0.5, color='cyan') -> px.scatter_3d:
        """
        Gets the interactive 3D scatter plot which can be visualized with the show() method
        :return: plotly scatter plot
        """
        scatter = px.scatter_3d(self.df,
                                x=self.df['x'],
                                y=self.df['y'],
                                z=self.df['z'],
                                opacity=0.5)
        #                        template = "plotly_dark")
        fig_traces = []
        for trace in range(len(scatter["data"])):
            fig_traces.append(scatter["data"][trace])
        for traces in fig_traces:
            scatter.append_trace(traces, row=1, col=1)

        scatter.update_traces(marker=dict(size=size,
                                          color=color),
                              selector=dict(mode='markers'))
        scatter.update_layout(height=400,
                              width=700,
                              template='plotly_dark')
        scatter.update_xaxes(showgrid=False)
        scatter.update_yaxes(showgrid=False)
        self.scatter = scatter
        return scatter


class Visualization2D:
    """
    Class for visualizing point cloud data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scatter = []
        self.get_2d_scatter()

    def get_2d_scatter(self) -> px.scatter:
        """
        Gets the interactive 2D scatter plot which can be visualized with the show() method
        :return: plotly scatter plot
        """
        if 'z' in self.df.columns:
            raise ValueError(f"Please use a 2D dataset")
        else:
            scatter = px.scatter(self.df,
                                 x=self.df['x'],
                                 y=self.df['y'],
                                 opacity=0.5)
            self.scatter = scatter
        return scatter


class loc2img:
    """
    Class for visualizing point cloud data.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scatter = []
        self.loc2img()

    def loc2img(self, pixel_size: int = 10):
        """
        Method taken from P. Kollmannsberger for reconstructing 2D localizations into an image
        :param pixel_size: size of the pixel for visualization
        :return: image
        """
        x = self.df['x']
        y = self.df['y']
        xbins = np.arange(x.min(), x.max()+1, pixel_size)
        ybins = np.arange(y.min(), y.max()+1, pixel_size)
        img, xe, ye = np.histogram2d(y, x, bins=(ybins, xbins))
        equalized = exposure.equalize_hist(img)
        return img, equalized


# TODO: probably delete this, not used
def save_plots(epoch, loss, pc_list, path, save=True):
    # fig = plt.figure()
    #
    detached = []
    for i, pc in enumerate(pc_list):
        if torch.is_tensor(pc):
            if pc.ndim == 3:
                pc = pc.permute(0, 2, 1)
                arr = pc.detach().cpu().numpy()
                arr = arr[0]
                detached.append(arr)
            else:
                arr = pc.detach().cpu().numpy()
                detached.append(arr)
        else:
            arr = pc
            detached.append(arr)
    #     ax = fig.add_subplot(1, 2, i + 1, projection='3d')
    #     x = arr[:, 0]
    #     y = arr[:, 1]
    #     z = arr[:, 2]
    #     ax.scatter(x, y, z)
    print_pc_from_filearray(detached, path=path, epoch=epoch, loss=loss, save=save)
    # Set the title with epoch number and loss


def print_pc(pc_list, permutation=False, save=False, path='/home/dim26fa/graphics'):
    """
    Plots one or multiple point clouds in a grid.
    :param pc_list: list containing point clouds as numpy arrays
    :param permutation: boolean to indicate if to permute the point cloud before plotting
    :param save: boolean to indicate if to save the plot
    :param path: path where to save the plot
    :return: None
    """
    import os
    fig = plt.figure()
    for i, pc in enumerate(pc_list):
        if torch.is_tensor(pc):
            if permutation:
                arr = pc.permute(0, 2, 1)
                arr = arr.detach().cpu().numpy()[0]
            else:
                arr = pc.detach().cpu().numpy()[0]
        else:
            if isinstance(pc, np.ndarray) and pc.ndim == 2:
                arr = pc
            else:
                arr = pc.transpose(0, 2, 1)[0]
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        ax.view_init(elev=30, azim=120)
        ax.scatter(x, y, z)
    if save:
        plt.savefig(os.path.join(path, 'pcs.png'))
    plt.show()


# Similar to function from readers.py
def _load_point_cloud(file_path):
    if file_path.endswith('.csv'):
        # Assuming the file format is CSV and contains three columns: x, y, z
        return np.genfromtxt(file_path, delimiter=',')
    elif file_path.endswith('.ply'):
        pc = o3d.io.read_point_cloud(file_path)
        return np.array(pc.points)


def _plot_point_cloud(point_clouds, overlay, path, epoch, loss, save=False):
    num_point_clouds = len(point_clouds)
    rows = int(np.ceil(np.sqrt(num_point_clouds)))
    cols = int(np.ceil(num_point_clouds / rows))

    colours = ['blue', 'orange', 'green']

    plt.style.use('dark_background')
    fig = plt.figure()
    fig.set_size_inches(19.20, 10.80)
    fig.patch.set_facecolor('black')
    if overlay:
        ax = fig.add_subplot(111, projection='3d')
        for i, pc in enumerate(point_clouds):
            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
            ax.scatter(x, y, z, zorder=i, c=colours[i])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    else:
        for i, pc in enumerate(point_clouds, 1):
            ax = fig.add_subplot(cols, rows, i, projection='3d')
            x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
            ax.scatter(x, y, z)
            ax.grid(False)
            ax.w_xaxis.pane.fill = False
            ax.w_yaxis.pane.fill = False
            ax.w_zaxis.pane.fill = False
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            # Set the color of the axis spines
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.zaxis.label.set_color('white')
            ax.tick_params(axis='both', colors='white')

    plt.tight_layout()


    #plt.suptitle(f"Epoch: {epoch}, Loss: {loss}")

    if save:
        # Create a directory to save the plots if it doesn't exist
        full_path = path + "/plots"
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        # Save the plot as a PNG with unique filename based on epoch and loss
        filename = f"/epoch_{epoch}_loss_{loss:.4f}.png"
        plt.savefig(full_path + filename)
        plt.close()
    else:
        plt.show()


def print_pc_from_filearray(point_clouds, path=None, epoch=0, loss=0, overlay=False, save=False):
    """
    Function to plot one or multiple point clouds in a grid from files or from an arrays.
    :param point_clouds: path or array
    :param path: path where to save the plot
    :param epoch: int indicating epoch number for which we plot
    :param loss: float indicating loss for the specific plot
    :param overlay: boolean indicating whether to overlay the point clouds from the list or plot separately
    :param save: boolean indicating whether to save the plot
    :return: None
    """
    if isinstance(point_clouds[0], str):
        # Load point clouds from file paths
        point_clouds = [_load_point_cloud(path) for path in point_clouds]

    if all(isinstance(pc, np.ndarray) for pc in point_clouds):
        # Plot point clouds from arrays
        _plot_point_cloud(point_clouds, overlay, path=path, epoch=epoch, loss=loss, save=save)
    else:
        raise ValueError("Invalid input. The function expects a list of file paths or a list of numpy arrays.")


def show_error_pc(gt, pred):
    """
    Plots the error between ground truth and predicted point cloud.
    :param gt: ground truth point cloud
    :param pred: predicted point cloud
    :return: None
    """
    if type(gt) != np.ndarray:
        gt = np.array(gt.points)
        pred = np.array(pred.points)
    from scipy.spatial import KDTree
    unique_pred = np.unique(pred, axis=0)
    tree = KDTree(gt)
    # Query the KD-tree for each source point to find its nearest neighbor in the target point cloud
    distances, _ = tree.query(unique_pred)
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


def overlay_pc(gt, pred):
    """
    Overlays two point clouds in red and blue colors
    :param gt: first point cloud
    :param pred: second point cloud
    :return: None
    """
    if type(gt) != np.ndarray:
        gt = np.array(gt.points) # assuming gt and pred are open3d point clouds
        pred = np.array(pred.points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], color='red', marker='o', label='pred')
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], color='blue', marker='o', label='gt')
    plt.show()


def o3d_plot(pc_list):
    """
    Uses open3d draw geometries to plot point clouds. It will be interactive.
    :param pc_list: list of point clouds to plot
    :return: None
    """
    o3d_list = []
    for pc in pc_list:
        # Check if the input is a path or an array
        if isinstance(pc, str) and os.path.isfile(pc):
            # If it's a path, read the point cloud file
            o3d_pc = o3d.io.read_point_cloud(pc)
        elif isinstance(pc, np.ndarray):
            # If it's an array, create an Open3D PointCloud object
            o3d_pc = o3d.geometry.PointCloud()
            o3d_pc.points = o3d.utility.Vector3dVector(pc)
        elif isinstance(pc, str) and os.path.isfile(pc) and pc.lower().endswith('.csv'):
            # If it's a path to a CSV file, read the CSV into a NumPy array
            point_data = pd.read_csv(pc).to_numpy()
            o3d_pc = o3d.geometry.PointCloud()
            o3d_pc.points = o3d.utility.Vector3dVector(point_data)
        else:
            print(f"Unsupported type or file not found: {pc}")
            continue
        # Generate random colors for each point cloud
        color = np.random.rand(3)
        o3d_pc.colors = o3d.utility.Vector3dVector(np.tile(color, (np.asarray(o3d_pc.points).shape[0], 1)))

        # Add to the list
        o3d_list.append(o3d_pc)

    # Visualize all point clouds together
    o3d.visualization.draw_geometries(o3d_list)


def hist_projection(point_cloud, grid_size=0.01, axis=2, cmap='hot'):
    """
    Projects a 3D point cloud onto a specified 2D plane/axis and plots the intensity image.

    Parameters:
    point_cloud: Nx3 numpy array
        The input 3D point cloud data.
    grid_size: float, default 0.01
        The size of each grid cell in the histogram.
    axis: int, default 2
        The axis along which the projection will be computed.
        0 = YZ plane, 1 = XZ plane, 2 = XY plane
    """
    # Determine the range of the data
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

    # Compute the number of bins along each axis
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    z_bins = int((z_max - z_min) / grid_size) + 1

    # Create the 3D histogram
    hist, edges = np.histogramdd(point_cloud, bins=[x_bins, y_bins, z_bins])

    # Project the 3D histogram onto the specified 2D plane by summing along the specified axis
    hist_2d = np.sum(hist, axis=axis)

    # Set axis labels and plot title based on the projection axis
    if axis == 0:  # YZ plane
        extent = [y_min, y_max, z_min, z_max]
        xlabel, ylabel = 'Y axis', 'Z axis'
        title = 'Projected Spatial Histogram (YZ Plane)'
    elif axis == 1:  # XZ plane
        extent = [x_min, x_max, z_min, z_max]
        xlabel, ylabel = 'X axis', 'Z axis'
        title = 'Projected Spatial Histogram (XZ Plane)'
    else:  # XY plane
        extent = [x_min, x_max, y_min, y_max]
        xlabel, ylabel = 'X axis', 'Y axis'
        title = 'Projected Spatial Histogram (XY Plane)'

    # Plot the 2D histogram as an intensity image
    plt.imshow(hist_2d.T, origin='lower', extent=extent, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Point Density')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def hist_projections(point_cloud, grid_size=0.01, cmap='hot', title=None, save=False):
    """
    Projects a 3D point cloud onto XY and XZ planes and plots the intensity images
    with consistent scaling and fixed bounding box size.

    :param point_cloud: The input 3D point cloud data. Expects Nx3 array.
    :param grid_size: The size of each grid cell in the histogram, default 0.01.
    :param cmap: The colormap to use for the intensity images, default 'hot'
    :param title: string, title for the plot. optional
    :param save: boolean deciding whether to save the plot as an image file. default False.
    :return: None
    """
    # Determine the range of the data
    x_min, y_min, z_min = np.min(point_cloud, axis=0)
    x_max, y_max, z_max = np.max(point_cloud, axis=0)

    # Find the maximum range across all dimensions
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Adjust the ranges to be centered and have the same size
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2

    x_min, x_max = x_center - max_range / 2, x_center + max_range / 2
    y_min, y_max = y_center - max_range / 2, y_center + max_range / 2
    z_min, z_max = z_center - max_range / 2, z_center + max_range / 2

    # Compute the number of bins
    num_bins = int(max_range / grid_size) + 1

    # Create the 3D histogram
    hist, _ = np.histogramdd(point_cloud, bins=(num_bins, num_bins, num_bins),
                             range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]])

    # Calculate projections
    hist_xy = np.sum(hist, axis=2)
    hist_xz = np.sum(hist, axis=1)

    # Find the maximum value for consistent scaling
    vmax = max(np.max(hist_xy), np.max(hist_xz))

    # Set up the figure with black background
    fig = plt.figure(figsize=(15, 7), facecolor='black')
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

    if title is not None:
        fig.suptitle(title, color='white', fontsize=16)

    # Common parameters for all subplots
    imshow_params = {
        'cmap': cmap,
        'interpolation': 'nearest',
        'vmin': 0,
        'vmax': vmax,
        'aspect': 'equal'
    }

    # XY Projection
    ax_xy = fig.add_subplot(gs[0, 0])
    im_xy = ax_xy.imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], **imshow_params)
    ax_xy.set_title('XY Projection', color='white')
    ax_xy.set_xlabel('X axis', color='white')
    ax_xy.set_ylabel('Y axis', color='white')

    # XZ Projection
    ax_xz = fig.add_subplot(gs[0, 1])
    im_xz = ax_xz.imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], **imshow_params)
    ax_xz.set_title('XZ Projection', color='white')
    ax_xz.set_xlabel('X axis', color='white')
    ax_xz.set_ylabel('Z axis', color='white')

    # Add colorbar
    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im_xy, cax=cax, label='Point Density')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    # Set background color for all subplots and adjust text color
    for ax in [ax_xy, ax_xz]:
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    plt.tight_layout()

    if save:
        directory = '/home/dim26fa/graphics/smlms/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, title + '.png'), dpi=300, bbox_inches='tight')

    plt.show()


def plot_centers_mass(gt, output, cluster_size):
    """
    Plots centers of mass of passed point clouds e.g. ground truth and output. Expects DataFrames.
    :param gt: point cloud 1
    :param output: point cloud 2
    :param cluster_size: size of cluster of HDBSCAN
    :return: None
    """
    import hdbscan
    from scipy.spatial.distance import euclidean

    output = output.copy()
    output = output.drop_duplicates(keep='first').reset_index(drop=True)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=2)
    cluster_labels_gt = clusterer.fit_predict(gt[['x', 'y', 'z']])
    gt['cluster_label'] = cluster_labels_gt
    centers_of_mass_gt = gt.groupby('cluster_label').mean().reset_index()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=2)
    cluster_labels_output = clusterer.fit_predict(output[['x', 'y', 'z']])
    output['cluster_label'] = cluster_labels_output
    centers_of_mass_output = output.groupby('cluster_label').mean().reset_index()

    # Calculate distances between corresponding points
    distances = []
    for i in range(min(len(centers_of_mass_gt), len(centers_of_mass_output))):
        gt_point = centers_of_mass_gt[['x', 'y', 'z']].iloc[i].values
        output_point = centers_of_mass_output[['x', 'y', 'z']].iloc[i].values
        distance = euclidean(gt_point, output_point)
        distances.append(distance)

        # Create 2D plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    views = [('x', 'y', ax1), ('x', 'z', ax2), ('y', 'z', ax3)]

    for dim1, dim2, ax in views:
        # Plot ground truth centers of mass
        ax.scatter(centers_of_mass_gt[dim1], centers_of_mass_gt[dim2],
                   c='blue', marker='o', s=100, label='Ground Truth')

        # Plot output centers of mass
        ax.scatter(centers_of_mass_output[dim1], centers_of_mass_output[dim2],
                   c='red', marker='^', s=100, label='Output')

        # Connect corresponding points with lines and add distance labels
        for i in range(min(len(centers_of_mass_gt), len(centers_of_mass_output))):
            ax.plot([centers_of_mass_gt[dim1].iloc[i], centers_of_mass_output[dim1].iloc[i]],
                    [centers_of_mass_gt[dim2].iloc[i], centers_of_mass_output[dim2].iloc[i]],
                    'k--', alpha=0.3)

            # Calculate midpoint for label placement
            mid_x = (centers_of_mass_gt[dim1].iloc[i] + centers_of_mass_output[dim1].iloc[i]) / 2
            mid_y = (centers_of_mass_gt[dim2].iloc[i] + centers_of_mass_output[dim2].iloc[i]) / 2

            # Add distance label
            ax.annotate(f'{distances[i]:.2f}', (mid_x, mid_y), xytext=(3, 3),
                        textcoords='offset points', fontsize=8, alpha=0.7)

        # Set labels and title
        ax.set_xlabel(dim1.upper())
        ax.set_ylabel(dim2.upper())
        ax.set_title(f'{dim1.upper()}{dim2.upper()} View')
        ax.legend()

    # Add overall title with average distance
    avg_distance = np.mean(distances)
    fig.suptitle(f'Centers of Mass: Ground Truth vs Output\nAverage Distance: {avg_distance:.2f}', fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_centers_mass_2d_with_matched_distances(gt, output, cluster_size, min_samples=2, save=False, path='/home/dim26fa/graphics/smlms'):
    """
    Plots centers of mass for e.g. ground truth and output with calculated distances between them. Expects DataFrames.
    :param gt: point cloud 1
    :param output: point cloud 2
    :param cluster_size: size of cluster for HDBSCAN
    :param min_samples: min_samples param for HDBSCAN, default 2
    :param save: boolean to indicate whether to save the figure
    :param path: path to where to save the figure if save is True
    :return:
    """
    import hdbscan
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    output = output.copy()
    output = output.drop_duplicates(keep='first').reset_index(drop=True)
    # Process ground truth data
    clusterer_gt = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=min_samples)
    cluster_labels_gt = clusterer_gt.fit_predict(gt[['x', 'y', 'z']])
    gt['cluster_label'] = cluster_labels_gt
    centers_of_mass_gt = gt.groupby('cluster_label').mean().reset_index()

    # Process output data
    clusterer_output = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=min_samples)
    cluster_labels_output = clusterer_output.fit_predict(output[['x', 'y', 'z']])
    output['cluster_label'] = cluster_labels_output
    centers_of_mass_output = output.groupby('cluster_label').mean().reset_index()

    # Match clusters using Hungarian algorithm
    cost_matrix = cdist(centers_of_mass_gt[['x', 'y', 'z']], centers_of_mass_output[['x', 'y', 'z']])
    gt_indices, output_indices = linear_sum_assignment(cost_matrix)

    # Calculate distances between matched points
    distances = cost_matrix[gt_indices, output_indices]

    # Create 2D plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    views = [('x', 'y', ax1), ('x', 'z', ax2), ('y', 'z', ax3)]

    for dim1, dim2, ax in views:
        # Plot all ground truth centers of mass
        ax.scatter(centers_of_mass_gt[dim1], centers_of_mass_gt[dim2],
                   c='blue', marker='o', s=100, label='Ground Truth')

        # Plot all output centers of mass
        ax.scatter(centers_of_mass_output[dim1], centers_of_mass_output[dim2],
                   c='red', marker='^', s=100, label='Output')

        # Connect and label only matched pairs
        for gt_idx, output_idx, distance in zip(gt_indices, output_indices, distances):
            gt_point = centers_of_mass_gt.iloc[gt_idx]
            output_point = centers_of_mass_output.iloc[output_idx]

            ax.plot([gt_point[dim1], output_point[dim1]],
                    [gt_point[dim2], output_point[dim2]],
                    'k--', alpha=0.3)

            # Calculate midpoint for label placement
            mid_x = (gt_point[dim1] + output_point[dim1]) / 2
            mid_y = (gt_point[dim2] + output_point[dim2]) / 2

            # Add distance label
            ax.annotate(f'{distance:.2f}', (mid_x, mid_y), xytext=(3, 3),
                        textcoords='offset points', fontsize=12, alpha=0.7)

        # Set labels and title
        ax.set_xlabel(dim1.upper())
        ax.set_ylabel(dim2.upper())
        ax.set_title(f'{dim1.upper()}{dim2.upper()} View')
        ax.legend()

    # Add overall title with average distance of matched pairs
    avg_distance = np.mean(distances)
    fig.suptitle(f'Centers of Mass: Ground Truth vs Output\n'
                 f'Average Distance of Matched Pairs: {avg_distance:.2f}\n'
                 f'GT Clusters: {len(centers_of_mass_gt)}, Output Clusters: {len(centers_of_mass_output)}',
                 fontsize=16)

    plt.tight_layout()

    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, 'centroids' + '.png'), dpi=300)


    plt.show()


def plot_projections_and_centroids(gt, output, cmap='hot', title=None, save=False):
    import hdbscan
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    # Process ground truth data
    clusterer_gt = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=2)
    cluster_labels_gt = clusterer_gt.fit_predict(gt[['x', 'y', 'z']])
    gt['cluster_label'] = cluster_labels_gt
    centers_of_mass_gt = gt.groupby('cluster_label').mean().reset_index()

    # Process output data
    clusterer_output = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=2)
    cluster_labels_output = clusterer_output.fit_predict(output[['x', 'y', 'z']])
    output['cluster_label'] = cluster_labels_output
    centers_of_mass_output = output.groupby('cluster_label').mean().reset_index()

    # Match clusters using Hungarian algorithm
    cost_matrix = cdist(centers_of_mass_gt[['x', 'y', 'z']], centers_of_mass_output[['x', 'y', 'z']])
    gt_indices, output_indices = linear_sum_assignment(cost_matrix)

    # Calculate distances between matched points
    distances = cost_matrix[gt_indices, output_indices]

    # Set up the figure with black background
    fig = plt.figure(figsize=(20, 14), facecolor='black')
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 0.05])

    if title is not None:
        fig.suptitle(title, color='white', fontsize=16)

    # Common parameters for all subplots
    imshow_params = {
        'cmap': cmap,
        'interpolation': 'nearest',
    }

    # Function to create histogram projection
    def create_projection(ax, data, dim1, dim2, title):
        hist, _, _ = np.histogram2d(data[dim1], data[dim2], bins=100)
        im = ax.imshow(hist.T, extent=[data[dim1].min(), data[dim1].max(),
                                       data[dim2].min(), data[dim2].max()],
                       origin='lower', **imshow_params)
        ax.set_title(title, color='white')
        ax.set_xlabel(f'{dim1} axis', color='white')
        ax.set_ylabel(f'{dim2} axis', color='white')
        return im

    # Create projections for ground truth
    ax_xy_gt = fig.add_subplot(gs[0, 0])
    im_xy_gt = create_projection(ax_xy_gt, gt, 'x', 'y', 'XY Projection (Ground Truth)')

    ax_xz_gt = fig.add_subplot(gs[0, 1])
    create_projection(ax_xz_gt, gt, 'x', 'z', 'XZ Projection (Ground Truth)')

    ax_yz_gt = fig.add_subplot(gs[0, 2])
    create_projection(ax_yz_gt, gt, 'y', 'z', 'YZ Projection (Ground Truth)')

    # Create projections for output
    ax_xy_out = fig.add_subplot(gs[1, 0])
    create_projection(ax_xy_out, output, 'x', 'y', 'XY Projection (Output)')

    ax_xz_out = fig.add_subplot(gs[1, 1])
    create_projection(ax_xz_out, output, 'x', 'z', 'XZ Projection (Output)')

    ax_yz_out = fig.add_subplot(gs[1, 2])
    im_yz_out = create_projection(ax_yz_out, output, 'y', 'z', 'YZ Projection (Output)')

    # Add colorbar
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(im_yz_out, cax=cax, label='Point Density')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    # Plot centroids and connections
    for gt_idx, output_idx, distance in zip(gt_indices, output_indices, distances):
        gt_point = centers_of_mass_gt.iloc[gt_idx]
        output_point = centers_of_mass_output.iloc[output_idx]

        # Plot on XY projections
        ax_xy_gt.plot(gt_point['x'], gt_point['y'], 'go', markersize=10, alpha=0.7)
        ax_xy_out.plot(output_point['x'], output_point['y'], 'yo', markersize=10, alpha=0.7)

        # Plot on XZ projections
        ax_xz_gt.plot(gt_point['x'], gt_point['z'], 'go', markersize=10, alpha=0.7)
        ax_xz_out.plot(output_point['x'], output_point['z'], 'yo', markersize=10, alpha=0.7)

        # Plot on YZ projections
        ax_yz_gt.plot(gt_point['y'], gt_point['z'], 'go', markersize=10, alpha=0.7)
        ax_yz_out.plot(output_point['y'], output_point['z'], 'yo', markersize=10, alpha=0.7)

    # Set background color for all subplots and adjust text color
    for ax in [ax_xy_gt, ax_xz_gt, ax_yz_gt, ax_xy_out, ax_xz_out, ax_yz_out]:
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    plt.tight_layout()

    if save:
        directory = '/home/dim26fa/graphics/smlms/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, title + '.png'), dpi=300, bbox_inches='tight')

    plt.show()

# TODO: check if this is needed
def plot_transparent(point_cloud, size=15,path='/home/dim26fa/graphics/smlms/'):
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    # Plot the points
    ax.scatter(x, y, z, c='teal', s=size, alpha=0.6)
    # Remove grid lines
    ax.grid(False)
    # Remove background planes (axes planes)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # Remove tick marks
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticks([])
    # Set axis limits (optional, adjust as needed)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_axis_off()

    plt.savefig(path + '3d_point_cloud.png', transparent=True, bbox_inches='tight',
                pad_inches=0)


def hist_projections_with_3d(point_cloud, grid_size=0.01, scatter_color='#1f77b4', density_cmap='hot', title=None, save=False):
    """
    Plots a 3D scatter of the point cloud in a single color and projects it onto XY and XZ planes,
    enclosed in a box without grid or background. Includes a colorbar for point density in 2D projections.

    Parameters:
    point_cloud: Nx3 numpy array
        The input 3D point cloud data.
    grid_size: float, default 0.01
        The size of each grid cell in the histogram.
    scatter_color: str, default '#1f77b4' (matplotlib default blue)
        The color to use for the 3D scatter plot.
    density_cmap: str, default 'hot'
        The colormap to use for the 2D projection intensity images.
    title: str, optional
        The title for the entire figure.
    save: bool, default False
        Whether to save the figure or not.
    """
    # Determine the range of the data
    x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

    # Compute the number of bins along each axis
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1
    z_bins = int((z_max - z_min) / grid_size) + 1

    # Create the 3D histogram
    hist, edges = np.histogramdd(point_cloud, bins=[x_bins, y_bins, z_bins])

    # Set up the figure with black background
    fig = plt.figure(figsize=(20, 7), facecolor='black')
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

    if title is not None:
        fig.suptitle(title, color='white', fontsize=16)

    # 3D Scatter Plot
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    scatter = ax_3d.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                            c=scatter_color, s=1)
    ax_3d.set_title('3D Scatter Plot', color='white')
    ax_3d.set_xlabel('X axis', color='white')
    ax_3d.set_ylabel('Y axis', color='white')
    ax_3d.set_zlabel('Z axis', color='white')
    ax_3d.tick_params(colors='white')

    # Remove background and grid, keep only the box
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('white')
    ax_3d.yaxis.pane.set_edgecolor('white')
    ax_3d.zaxis.pane.set_edgecolor('white')
    ax_3d.grid(False)

    # XY Projection (axis=2)
    ax_xy = fig.add_subplot(gs[0, 1])
    hist_xy = np.sum(hist, axis=2)
    im_xy = ax_xy.imshow(hist_xy.T, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap=density_cmap)
    ax_xy.set_title('XY Projection', color='white')
    ax_xy.set_xlabel('X axis', color='white')
    ax_xy.set_ylabel('Y axis', color='white')

    # XZ Projection (axis=1)
    ax_xz = fig.add_subplot(gs[0, 2])
    hist_xz = np.sum(hist, axis=1)
    im_xz = ax_xz.imshow(hist_xz.T, extent=[x_min, x_max, z_min, z_max], origin='lower', cmap=density_cmap)
    ax_xz.set_title('XZ Projection', color='white')
    ax_xz.set_xlabel('X axis', color='white')
    ax_xz.set_ylabel('Z axis', color='white')

    # Add colorbar for the 2D projections (point density)
    cax_density = fig.add_subplot(gs[0, 3])
    cbar_density = fig.colorbar(im_xy, cax=cax_density, label='Point Density')
    cbar_density.ax.yaxis.label.set_color('white')
    cbar_density.ax.tick_params(colors='white')

    # Set background color for all subplots and adjust text color
    for ax in [ax_xy, ax_xz]:
        ax.set_facecolor('none')  # Transparent background
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.grid(False)  # Remove grid

    plt.tight_layout()

    if save:
        directory = '/home/dim26fa/graphics/smlms/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, title + '.png'), dpi=300, transparent=True)

    plt.show()


def create_confusion_matrix(true_labels, predictions, class_names=['Class 0', 'Class 1']):
    """
    Plots confusion matrix between true labels and predictions.
    :param true_labels: array containing true labels
    :param predictions: array containing predicted labels
    :param class_names: list containing the class names.
    :return:
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    # Convert two-column predictions to class predictions
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)

    # Convert two-column true labels to single column if necessary
    if true_labels.ndim == 2:
        true_labels = np.argmax(true_labels, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def get_classification_metrics(true_labels, predicted_labels, plot=False):
    """
    Gets classification metrics. Plots if plot=True and returns dict.
    :param true_labels: array containing true labels
    :param predicted_labels: array containing predicted labels
    :param plot: boolean to plot confusion matrix
    :return: dictionary containig the metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Compute accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Compute precision, recall, and F1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    if plot:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [metrics[m] for m in metrics_to_plot]

        plt.figure(figsize=(10, 6))
        plt.bar(metrics_to_plot, values)
        plt.title('Classification Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
        plt.show()

    return metrics


def plot_single_3d(point_clouds, background='black', point_sizes=None, figsize=(10, 8), alphas=None):
    """
    Plot one or more 3D point clouds with a customizable background and fully visible bounding box.

    Parameters:
    point_clouds (list): A list of numpy arrays, each of shape (n, 3) representing a point cloud
    background (str): Background color, either 'black' or 'white'
    point_sizes (list): List of point sizes for each cloud
    figsize (tuple): Figure size in inches
    alphas (list): List of alpha values for each cloud
    """
    if background not in ['black', 'white']:
        raise ValueError("Background must be either 'black' or 'white'")

    fg_color = 'white' if background == 'black' else 'black'

    if point_sizes is None:
        point_sizes = [5, 15]
    if alphas is None:
        alphas = [0.5, 1.0]

    fig = plt.figure(figsize=figsize, facecolor=background)
    ax = fig.add_subplot(111, projection='3d')

    # Define color schemes for each background
    color_schemes = {
        'black': ['teal', 'white', 'yellow', 'red', 'blue', 'cyan', 'magenta'],
        'white': ['teal', 'black', 'orange', 'red', 'blue', 'green', 'purple']
    }
    colors = color_schemes[background]

    all_points = np.concatenate(point_clouds, axis=0)

    for i, points in enumerate(point_clouds):
        points = np.array(points)
        if points.shape[1] != 3:
            raise ValueError(f"Input array {i} must have 3 columns (x, y, z coordinates)")
        color = colors[i % len(colors)]
        psize = point_sizes[i % len(point_sizes)]
        alpha = alphas[i % len(alphas)]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=psize, alpha=alpha, label=f'Cloud {i + 1}')

    # Customize the plot
    ax.set_facecolor(background)
    ax.grid(False)
    ax.set_xlabel('X', color=fg_color)
    ax.set_ylabel('Y', color=fg_color)
    ax.set_zlabel('Z', color=fg_color)

    # Set colors for axes, labels, and ticks
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_color(fg_color)
        axis.set_tick_params(colors=fg_color)

    # Remove panes and their edges
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor('none')

    # Adjust plot limits to ensure full bounding box visibility
    ax.set_xlim(all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
    ax.set_ylim(all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)
    ax.set_zlim(all_points[:, 2].min() - 1, all_points[:, 2].max() + 1)

    # Add legend if there are multiple point clouds
    if len(point_clouds) > 1:
        ax.legend(facecolor=background, edgecolor=fg_color, labelcolor=fg_color)

    # Remove white space around the plot
    plt.tight_layout()

    # Adjust the axes aspect ratio to be equal
    ax.set_box_aspect((1, 1, 1))

    plt.show()

def gaussian(x, a, mu, sigma):
    return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def plot_z_histogram(point_clouds, bins=50, figsize=(10, 6), save=False, background='black'):
    """
    Plot a histogram of z-coordinates from one or more point clouds with a Gaussian fit.

    Parameters:
    point_clouds (list): A list of numpy arrays, each of shape (n, 3) representing a point cloud
    bins (int): Number of bins for the histogram
    figsize (tuple): Figure size in inches
    save (bool): Whether to save the figure as a PNG
    background (str): 'black' or 'white', setting the background color of the plot
    """
    from scipy.signal import find_peaks
    from sklearn.mixture import GaussianMixture
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # Set colors based on background choice
    if background == 'black':
        bg_color = 'black'
        fg_color = 'white'
        hist_color = 'teal'
        fit_color = 'orange'
        line_color = 'white'
    else:
        bg_color = 'white'
        fg_color = 'black'
        hist_color = 'darkblue'
        fit_color = 'darkred'
        line_color = 'black'

    fig, ax = plt.subplots(figsize=figsize, facecolor=bg_color)
    ax.set_facecolor(bg_color)

    all_z = np.concatenate([cloud[:, 2] for cloud in point_clouds])

    # Plot histogram
    counts, bins, _ = ax.hist(all_z, bins=bins, color=hist_color, alpha=0.7, density=True, label='Histogram')

    # Fit two-Gaussian mixture
    gm = GaussianMixture(n_components=2, random_state=0).fit(all_z.reshape(-1, 1))

    # Generate points for plotting the mixture
    x_fit = np.linspace(bins.min(), bins.max(), 1000).reshape(-1, 1)
    y_fit = np.exp(gm.score_samples(x_fit))

    # Plot Gaussian mixture fit
    ax.plot(x_fit, y_fit, color=fit_color, linewidth=2, label='Gaussian mixture fit')

    # Customize plot
    ax.set_xlabel('z axis', color=fg_color)
    ax.set_ylabel('Counts', color=fg_color)
    ax.tick_params(axis='x', colors=fg_color)
    ax.tick_params(axis='y', colors=fg_color)
    for spine in ax.spines.values():
        spine.set_color(fg_color)

    # Add legend
    ax.legend(facecolor=bg_color, edgecolor=fg_color, labelcolor=fg_color)

    # Find peaks in the fitted mixture
    peaks, _ = find_peaks(y_fit.flatten(), height=max(y_fit) / 4, distance=len(x_fit) // 10)
    peak_x = x_fit[peaks].flatten()

    # Add text for peak distance if there are two distinct peaks
    if len(peak_x) == 2:
        peak_distance = abs(peak_x[1] - peak_x[0])
        ax.text(0.7, 0.95, f'~{peak_distance:.0f}', transform=ax.transAxes,
                color=fit_color, verticalalignment='top')

        # Add vertical lines at peak positions
        for px in peak_x:
            ax.axvline(px, color=line_color, linestyle='--', alpha=0.5)

    plt.title('Histogram of z-coordinates', color=fg_color)
    plt.tight_layout()

    if save:
        directory = '/home/dim26fa/graphics/smlms/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, 'histogram' + '.png'), dpi=300, bbox_inches='tight')
    plt.show()


def hist_projections_with_model(point_cloud, model_structure, grid_size=0.01, cmap='hot', title=None, save=False,
                                background='black'):
    """
    Projects a 3D point cloud and model structure onto XY and XZ planes and plots the intensity images
    with consistent scaling and fixed bounding box size.
    :param point_cloud: Nx3 numpy array. The input 3D point cloud data.
    :param model_structure: Mx3 numpy array. The input 3D model structure data.
    :param grid_size: float, default 0.01. The size of each grid cell in the histogram.
    :param cmap: str, default 'hot'. The colormap to use for the intensity images.
    :param title: str, optional. The title for the plot.
    :param save: bool, default False. Whether to save the plot as an image file.
    :param background: str, default 'black'. 'black' or 'white' background color for the plot.
    """

    # Set colors based on background choice
    if background == 'black':
        bg_color = 'black'
        fg_color = 'white'
        scatter_color = 'white'
    else:
        bg_color = 'white'
        fg_color = 'black'
        scatter_color = 'white'

    # Combine point cloud and model structure for range calculation
    all_points = np.vstack((point_cloud, model_structure))

    # Determine the range of the data
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)

    # Find the maximum range across all dimensions
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Adjust the ranges to be centered and have the same size
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    x_min, x_max = x_center - max_range / 2, x_center + max_range / 2
    y_min, y_max = y_center - max_range / 2, y_center + max_range / 2
    z_min, z_max = z_center - max_range / 2, z_center + max_range / 2

    # Compute the number of bins
    num_bins = int(max_range / grid_size) + 1

    # Create the 3D histogram for point cloud
    hist, _ = np.histogramdd(point_cloud, bins=(num_bins, num_bins, num_bins),
                             range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]])

    # Calculate projections for point cloud
    hist_xy = np.sum(hist, axis=2)
    hist_xz = np.sum(hist, axis=1)

    # Find the maximum value for consistent scaling
    vmax = max(np.max(hist_xy), np.max(hist_xz))

    # Set up the figure with the selected background color
    fig = plt.figure(figsize=(15, 7), facecolor=bg_color)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
    if title is not None:
        fig.suptitle(title, color=fg_color, fontsize=16)

    # Common parameters for all subplots
    imshow_params = {
        'cmap': cmap,
        'interpolation': 'nearest',
        'vmin': 0,
        'vmax': vmax,
        'aspect': 'equal',
        'alpha': 0.4  # Set alpha for point cloud visualization
    }

    # XY Projection
    ax_xy = fig.add_subplot(gs[0, 0])
    im_xy = ax_xy.imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], **imshow_params)
    ax_xy.scatter(model_structure[:, 0], model_structure[:, 1], c=scatter_color, s=15,
                  alpha=1)  # Add model structure
    ax_xy.set_title('XY Projection', color=fg_color)
    ax_xy.set_xlabel('X axis', color=fg_color)
    ax_xy.set_ylabel('Y axis', color=fg_color)

    # XZ Projection
    ax_xz = fig.add_subplot(gs[0, 1])
    im_xz = ax_xz.imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], **imshow_params)
    ax_xz.scatter(model_structure[:, 0], model_structure[:, 2], c=scatter_color, s=15,
                  alpha=1)  # Add model structure
    ax_xz.set_title('XZ Projection', color=fg_color)
    ax_xz.set_xlabel('X axis', color=fg_color)
    ax_xz.set_ylabel('Z axis', color=fg_color)

    # Set background color for all subplots and adjust text color
    for ax in [ax_xy, ax_xz]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=fg_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg_color)

    plt.tight_layout()

    if save:
        directory = '/home/dim26fa/graphics/smlms/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, title + '.png'), dpi=300, bbox_inches='tight')

    plt.show()


def hist_projections_with_model_and_centroid(point_cloud, model_structure, cluster_size=5, min_samples=20, grid_size=0.01, cmap='hot', title=None, save=False, background='black'):
    """
    Projects a 3D point cloud and model structure onto XY and XZ planes, plots the intensity images
    with consistent scaling and fixed bounding box size, and includes cluster centroids.
    :param point_cloud: Nx3 numpy array. The input 3D point cloud data.
    :param model_structure: Mx3 numpy array. The input 3D model structure data.
    :param cluster_size: int, default 5. The minimum cluster size for HDBSCAN.
    :param min_samples: int, optional. The number of samples in a neighborhood for a point to be considered a core point.
    :param grid_size: float, default 0.01. The size of each grid cell in the histogram.
    :param cmap: str, default 'hot'. The colormap to use for the intensity images.
    :param title: str, optional. The title for the plot.
    :param save: bool, default False. Whether to save the plot as an image file.
    :param background: str, default 'black'. 'black' or 'white' background color for the plot.
    """
    import hdbscan

    # Set colors based on background choice
    if background == 'black':
        bg_color = 'black'
        fg_color = 'white'
        scatter_color = 'white'
        centroid_color = 'teal'
    else:
        bg_color = 'white'
        fg_color = 'black'
        scatter_color = 'white'
        centroid_color = 'teal'

    # Convert point cloud to DataFrame
    df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])

    # Perform clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_size, min_samples=min_samples)
    df['cluster_label'] = clusterer.fit_predict(df[['x', 'y', 'z']])
    df_filtered = df[df['cluster_label'] != -1]
    # Calculate centroids (centers of mass) for each cluster
    centers_of_mass = df_filtered.groupby('cluster_label').mean().reset_index()

    # Combine point cloud and model structure for range calculation
    all_points = np.vstack((point_cloud, model_structure))

    # Determine the range of the data
    x_min, y_min, z_min = np.min(all_points, axis=0)
    x_max, y_max, z_max = np.max(all_points, axis=0)

    # Find the maximum range across all dimensions
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)

    # Adjust the ranges to be centered and have the same size
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    x_min, x_max = x_center - max_range / 2, x_center + max_range / 2
    y_min, y_max = y_center - max_range / 2, y_center + max_range / 2
    z_min, z_max = z_center - max_range / 2, z_center + max_range / 2

    # Compute the number of bins
    num_bins = int(max_range / grid_size) + 1

    # Create the 3D histogram for point cloud
    hist, _ = np.histogramdd(point_cloud, bins=(num_bins, num_bins, num_bins),
                             range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]])

    # Calculate projections for point cloud
    hist_xy = np.sum(hist, axis=2)
    hist_xz = np.sum(hist, axis=1)

    # Find the maximum value for consistent scaling
    vmax = max(np.max(hist_xy), np.max(hist_xz))

    # Set up the figure with the selected background color
    fig = plt.figure(figsize=(15, 7), facecolor=bg_color)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
    if title is not None:
        fig.suptitle(title, color=fg_color, fontsize=16)

    # Common parameters for all subplots
    imshow_params = {
        'cmap': cmap,
        'interpolation': 'nearest',
        'vmin': 0,
        'vmax': vmax,
        'aspect': 'equal',
        'alpha': 0.4  # Set alpha for point cloud visualization
    }

    # XY Projection
    ax_xy = fig.add_subplot(gs[0, 0])
    im_xy = ax_xy.imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], **imshow_params)
    ax_xy.scatter(model_structure[:, 0], model_structure[:, 1], c=scatter_color, s=80, alpha=1, edgecolors=fg_color, label='model')  # Add model structure
    if len(centers_of_mass) > 0:
        ax_xy.scatter(centers_of_mass['x'], centers_of_mass['y'], c=centroid_color, s=70, marker='s', edgecolors=fg_color, label='calculated')  # Add centroids
    ax_xy.set_title('XY Projection', color=fg_color)
    ax_xy.set_xlabel('X axis', color=fg_color)
    ax_xy.set_ylabel('Y axis', color=fg_color)
    plt.legend(fontsize=14)

    # XZ Projection
    ax_xz = fig.add_subplot(gs[0, 1])
    im_xz = ax_xz.imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], **imshow_params)
    ax_xz.scatter(model_structure[:, 0], model_structure[:, 2], c=scatter_color, s=80, alpha=1, edgecolors=fg_color, label='model')  # Add model structure
    if len(centers_of_mass) > 0:
        ax_xz.scatter(centers_of_mass['x'], centers_of_mass['z'], c=centroid_color, s=70, marker='s', edgecolors=fg_color, label='calculated')  # Add centroids
    ax_xz.set_title('XZ Projection', color=fg_color)
    ax_xz.set_xlabel('X axis', color=fg_color)
    ax_xz.set_ylabel('Z axis', color=fg_color)

    # Set background color for all subplots and adjust text color
    for ax in [ax_xy, ax_xz]:
        ax.set_facecolor(bg_color)
        ax.tick_params(colors=fg_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(fg_color)

    plt.tight_layout()

    if save:
        directory = '/home/dim26fa/graphics/smlms/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, title + '.png'), dpi=300, bbox_inches='tight')

    plt.show()

    return centers_of_mass


# TODO: these functions have to be checked. not used.
def calculate_tetrahedron_distances(cloud1, cloud2):
    from itertools import permutations
    cloud1 = np.array(cloud1)
    cloud2 = np.array(cloud2)

    all_permutations = list(permutations(cloud2))

    min_total_distance = float('inf')
    best_matching = None

    for perm in all_permutations:
        perm_cloud = np.array(perm)
        distances = np.linalg.norm(cloud1 - perm_cloud, axis=1)
        total_distance = np.sum(distances)

        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_matching = list(zip(cloud1, perm, distances))

    return best_matching, min_total_distance


def calculate_adaptive_tetrahedron_distances(cloud1, cloud2):
    from itertools import permutations, combinations
    cloud1 = np.array(cloud1)
    cloud2 = np.array(cloud2)

    # Determine the smaller cloud
    if len(cloud1) < len(cloud2):
        smaller_cloud, larger_cloud = cloud1, cloud2
    else:
        smaller_cloud, larger_cloud = cloud2, cloud1

    min_total_distance = float('inf')
    best_matching = None

    # Generate all possible subsets of the larger cloud that match the size of the smaller cloud
    for subset in combinations(larger_cloud, len(smaller_cloud)):
        subset = np.array(subset)

        # Generate all permutations of the subset
        for perm in permutations(subset):
            perm_cloud = np.array(perm)

            # Calculate distances
            if len(cloud1) < len(cloud2):
                distances = np.linalg.norm(smaller_cloud - perm_cloud, axis=1)
            else:
                distances = np.linalg.norm(perm_cloud - smaller_cloud, axis=1)

            total_distance = np.sum(distances)

            if total_distance < min_total_distance:
                min_total_distance = total_distance
                if len(cloud1) < len(cloud2):
                    best_matching = list(zip(smaller_cloud, perm, distances))
                else:
                    best_matching = list(zip(perm, smaller_cloud, distances))

    return best_matching, min_total_distance