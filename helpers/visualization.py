import plotly.express as px
import open3d as o3d
import pandas as pd
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
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


def print_pc(pc_list, permutation=False):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for i, pc in enumerate(pc_list):
        if torch.is_tensor(pc):
            #arr = pc.permute(0, 2, 1)
            arr = pc.detach().cpu().numpy()
            arr = arr[0]
        else:
            if pc.ndim == 2:
                arr = pc
            else:
                arr = pc.transpose(0, 2, 1)
                arr = arr[0]
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        ax.scatter(x, y, z)
    # fig.title('cd is ' + str(cd))
    plt.show()


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



def print_pc(pc_list, permutation=False):
    import matplotlib.pyplot as plt
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
        ax.scatter(x, y, z)
    plt.show()


def _load_point_cloud(file_path):
    # Assuming the file format is CSV and contains three columns: x, y, z
    return np.genfromtxt(file_path, delimiter=',')


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
    plt.show()

    plt.suptitle(f"Epoch: {epoch}, Loss: {loss}")

    if save:
        # Create a directory to save the plots if it doesn't exist
        full_path = path + "/plots"
        if not os.path.exists(full_path):
            os.makedirs(full_path)

        # Save the plot as a PNG with unique filename based on epoch and loss
        filename = f"/epoch_{epoch}_loss_{loss:.4f}.png"
        plt.savefig(full_path + filename)
        plt.close()


def print_pc_from_filearray(point_clouds, path=None, epoch=0, loss=0, overlay=False, save=False):
    if isinstance(point_clouds[0], str):
        # Load point clouds from file paths
        point_clouds = [_load_point_cloud(path) for path in point_clouds]

    if all(isinstance(pc, np.ndarray) for pc in point_clouds):
        # Plot point clouds from arrays
        _plot_point_cloud(point_clouds, overlay, path=path, epoch=epoch, loss=loss, save=save)
    else:
        raise ValueError("Invalid input. The function expects a list of file paths or a list of numpy arrays.")


def show_error_pc(gt, pred):
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
    if type(gt) != np.ndarray:
        gt = np.array(gt.points) # assuming gt and pred are open3d point clouds
        pred = np.array(pred.points)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], color='red', marker='o', label='pred')
    ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], color='blue', marker='o', label='gt')
    plt.show()