import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import seaborn as sns
from scipy.stats import rayleigh, norm, maxwell
from scipy.ndimage import gaussian_filter


def plot_xy_xz_histograms(data, grid_size=0.01, cmap='hot', save_to=None):
    """
    Plot XY and XZ histogram projections from a 3D point cloud and optionally save the plot.

    Parameters:
    - data (str | pd.DataFrame | np.ndarray): Can be a file path to a .csv file, a DataFrame, or a NumPy array.
    - grid_size (float): Size of the grid cells for the histogram bins.
    - save_to (str | None): File path to save the plot. If None, the plot is shown instead.
    """
    try:
        # Handle different input types (file, DataFrame, or array)
        if isinstance(data, str):  # If a file path is provided
            data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):  # If a DataFrame is directly provided
            pass
        elif isinstance(data, np.ndarray):  # If a NumPy array is provided
            if data.shape[1] != 3:
                raise ValueError("The input array must have 3 columns representing x, y, and z coordinates.")
            data = pd.DataFrame(data, columns=['x', 'y', 'z'])
        else:
            raise ValueError("Data must be either a file path, a Pandas DataFrame, or a NumPy array.")

        # Ensure the DataFrame contains the required columns
        if not all(col in data.columns for col in ['x', 'y', 'z']):
            raise ValueError("The data must contain columns: 'x', 'y', and 'z'")

        # Convert the data to a 3D point cloud
        point_cloud = data[['x', 'y', 'z']].to_numpy()

        # Determine the range and bins for the histogram
        x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
        y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
        z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

        x_bins = int((x_max - x_min) / grid_size) + 1
        y_bins = int((y_max - y_min) / grid_size) + 1
        z_bins = int((z_max - z_min) / grid_size) + 1

        # Compute the 3D histogram
        hist, edges = np.histogramdd(point_cloud, bins=[x_bins, y_bins, z_bins])

        # Project the 3D histogram onto the XY and XZ planes
        hist_xy = np.sum(hist, axis=2)  # Projection onto XY plane
        hist_xz = np.sum(hist, axis=1)  # Projection onto XZ plane

        hist_xy = gaussian_filter(hist_xy, sigma=1)
        hist_xz = gaussian_filter(hist_xz, sigma=1)

        # Visualize the XY and XZ projections
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # XY Projection
        axs[0].imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto', cmap=cmap)
        axs[0].set_title('XY Projection')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        #axs[0].set_xticks([])  # Remove ticks
        #axs[0].set_yticks([])

        # XZ Projection
        axs[1].imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], aspect='auto', cmap=cmap)
        axs[1].set_title('XZ Projection')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Z')
        #axs[1].set_xticks([])  # Remove ticks
        #axs[1].set_yticks([])

        plt.tight_layout()

        # Save or show the plot
        if save_to:
            plt.savefig(save_to, dpi=600)  # Save the plot to the specified file path
            print(f"Plot saved to: {save_to}")
            plt.show()
        else:
            plt.show()  # Show the plot interactively

        plt.close(fig)  # Close the figure after saving or displaying it

    except FileNotFoundError:
        print(f"Error: File '{data}' not found.")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_xy_xz_histograms(data, grid_size=0.01, model_structure=None, cmap='hot', save_to=None):
    """
    Plot XY and XZ histogram projections from a 3D point cloud and overlay model_structure if provided.

    Parameters:
    - data (str | pd.DataFrame | np.ndarray): File path, DataFrame, or NumPy array of shape Nx3.
    - grid_size (float): Size of the histogram bins.
    - model_structure (np.ndarray | None): Optional 4x3 array to overlay on the projections.
    - cmap (str): Color map for the histogram.
    - save_to (str | None): File path to save the plot. If None, the plot is shown.
    """
    try:
        # Handle input data
        if isinstance(data, str):
            data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            pass
        elif isinstance(data, np.ndarray):
            if data.shape[1] != 3:
                raise ValueError("Input array must have 3 columns (x, y, z).")
            data = pd.DataFrame(data, columns=['x', 'y', 'z'])
        else:
            raise ValueError("Data must be a file path, DataFrame, or NumPy array.")

        if not all(col in data.columns for col in ['x', 'y', 'z']):
            raise ValueError("Data must contain 'x', 'y', and 'z' columns.")

        # Extract point cloud
        point_cloud = data[['x', 'y', 'z']].to_numpy()

        # Define bounds and bins
        x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
        y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
        z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

        # Update min/max to include both point cloud and model_structure
        if model_structure is not None and model_structure.shape == (4, 3):
            x_min = min(x_min, np.min(model_structure[:, 0]))
            x_max = max(x_max, np.max(model_structure[:, 0]))
            y_min = min(y_min, np.min(model_structure[:, 1]))
            y_max = max(y_max, np.max(model_structure[:, 1]))
            z_min = min(z_min, np.min(model_structure[:, 2]))
            z_max = max(z_max, np.max(model_structure[:, 2]))

        x_bins = int((x_max - x_min) / grid_size) + 1
        y_bins = int((y_max - y_min) / grid_size) + 1
        z_bins = int((z_max - z_min) / grid_size) + 1

        # 3D histogram and projections
        hist, _ = np.histogramdd(point_cloud, bins=[x_bins, y_bins, z_bins])
        hist_xy = gaussian_filter(np.sum(hist, axis=2), sigma=1)
        hist_xz = gaussian_filter(np.sum(hist, axis=1), sigma=1)

        base_alpha = 0.5 if model_structure is not None else 1.0

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # XY Projection
        axs[0].imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto', cmap=cmap, alpha=base_alpha)
        axs[0].set_title('XY Projection')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')

        # Overlay model structure on XY
        if model_structure is not None:
            axs[0].scatter(model_structure[:, 0], model_structure[:, 1], c='white', edgecolor='white',
                           s=30, label='Model Structure')
            axs[0].legend()

        # XZ Projection
        axs[1].imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], aspect='auto', cmap=cmap, alpha=base_alpha)
        axs[1].set_title('XZ Projection')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Z')

        # Overlay model structure on XZ
        if model_structure is not None:
            axs[1].scatter(model_structure[:, 0], model_structure[:, 2], c='white', edgecolor='white',
                           s=30, label='Model Structure')
            axs[1].legend()

        plt.tight_layout()
        if save_to:
            plt.savefig(save_to, dpi=600)
            print(f"Plot saved to: {save_to}")
        else:
            plt.show()

        plt.close(fig)

    except Exception as e:
        print(f"Error: {e}")


def plot_multiple_projections(data_list, grid_size=0.01, sigma=1, cmap='hot', save_to=None):
    """
    Plot multiple sets of XY and XZ histogram projections, stacking the plots in a grid.

    Parameters:
    - data_list (list): A list of datasets (can be file paths, DataFrames, or NumPy arrays).
    - grid_size (float): Size of the grid cells for the histogram bins.
    - save_to (str | None): File path to save the plot. If None, the plot is shown instead.
    - dpi (int): Dots per inch for saving the figure. Default is 100.
    """
    try:
        num_datasets = len(data_list)
        if num_datasets == 0:
            raise ValueError("The data list must contain at least one dataset.")

        # Create the figure with subplots: Two subplots per dataset (XY, XZ projections)
        fig, axs = plt.subplots(num_datasets, 2, figsize=(12, 6 * num_datasets))  # Two plots per row (XY and XZ)

        # Handle the case where there's only one dataset (axs won't be a 2D array)
        if num_datasets == 1:
            axs = [axs]

        for i, data in enumerate(data_list):
            # Load the data based on its type (file path, DataFrame, or NumPy array)
            if isinstance(data, str):  # If it's a file path
                data = pd.read_csv(data)
            elif isinstance(data, pd.DataFrame):  # If it's a DataFrame
                pass
            elif isinstance(data, np.ndarray):  # If it's a NumPy array
                if data.shape[1] != 3:
                    raise ValueError("Each input array must have 3 columns representing x, y, and z coordinates.")
                data = pd.DataFrame(data, columns=['x', 'y', 'z'])
            else:
                raise ValueError("Each dataset must be a file path, Pandas DataFrame, or NumPy array.")

            # Ensure the DataFrame contains the required columns
            if not all(col in data.columns for col in ['x', 'y', 'z']):
                raise ValueError("Each DataFrame must contain the columns: 'x', 'y', and 'z'")

            # Convert DataFrame to a NumPy array
            point_cloud = data[['x', 'y', 'z']].to_numpy()

            # Determine the range and bins for the histogram
            x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
            y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
            z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

            x_bins = int((x_max - x_min) / grid_size) + 1
            y_bins = int((y_max - y_min) / grid_size) + 1
            z_bins = int((z_max - z_min) / grid_size) + 1

            # Compute the 3D histogram
            hist, edges = np.histogramdd(point_cloud, bins=[x_bins, y_bins, z_bins])

            # Project the 3D histogram onto the XY and XZ planes
            hist_xy = np.sum(hist, axis=2)  # Projection onto XY plane
            hist_xz = np.sum(hist, axis=1)  # Projection onto XZ plane

            hist_xy = gaussian_filter(hist_xy, sigma=sigma)  # Adjust sigma for smoothness
            hist_xz = gaussian_filter(hist_xz, sigma=sigma)

            # Plot the XY projection
            axs[i][0].imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto', cmap=cmap)
            axs[i][0].set_title('XY Projection')
            axs[i][0].set_xlabel('X')
            axs[i][0].set_ylabel('Y')

            # Plot the XZ projection
            axs[i][1].imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], aspect='auto', cmap=cmap)
            axs[i][1].set_title('XZ Projection')
            axs[i][1].set_xlabel('X')
            axs[i][1].set_ylabel('Z')

        # Adjust layout to avoid overlapping titles/labels
        plt.tight_layout()

        # Save or show the grid of plots
        if save_to:
            plt.savefig(save_to, dpi=600)
            print(f"Multi-dataset plots saved to: {save_to}")
            plt.show()
        else:
            plt.show()

        plt.close(fig)  # Close the figure to free memory

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def plot_multiple_projections_grid(data_list, grid_size=0.01, sigma=1, cmap='hot', cols=2, save_to=None):
    """
    Plot multiple sets of XY and XZ histogram projections in a grid with 4 columns.

    Parameters:
    - data_list (list): A list of datasets (can be file paths, DataFrames, or NumPy arrays).
    - grid_size (float): Size of the grid cells for the histogram bins.
    - sigma (float): Standard deviation for Gaussian smoothing.
    - cmap (str): Colormap for the projections.
    - save_to (str | None): File path to save the plot. If None, the plot is shown instead.
    """
    try:
        num_datasets = len(data_list)
        if num_datasets == 0:
            raise ValueError("The data list must contain at least one dataset.")

        # Define grid layout
        rows = (num_datasets + cols - 1) // cols  # Calculate required rows

        fig, axs = plt.subplots(rows, cols * 2, figsize=(6 * cols, 4 * rows))  # Adjust figure size

        # Flatten axs in case it's 2D, to make indexing easier
        axs = axs.reshape(-1, 2)

        for i, data in enumerate(data_list):
            # Load the data
            if isinstance(data, str):
                data = pd.read_csv(data)
            elif isinstance(data, pd.DataFrame):
                pass
            elif isinstance(data, np.ndarray):
                if data.shape[1] != 3:
                    raise ValueError("Each input array must have 3 columns representing x, y, and z coordinates.")
                data = pd.DataFrame(data, columns=['x', 'y', 'z'])
            else:
                raise ValueError("Each dataset must be a file path, Pandas DataFrame, or NumPy array.")

            # Ensure DataFrame has required columns
            if not all(col in data.columns for col in ['x', 'y', 'z']):
                raise ValueError("Each DataFrame must contain the columns: 'x', 'y', and 'z'")

            # Convert DataFrame to NumPy array
            point_cloud = data[['x', 'y', 'z']].to_numpy()

            # Determine histogram bins
            x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
            y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
            z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

            x_bins = int((x_max - x_min) / grid_size) + 1
            y_bins = int((y_max - y_min) / grid_size) + 1
            z_bins = int((z_max - z_min) / grid_size) + 1

            # Compute the 3D histogram
            hist, edges = np.histogramdd(point_cloud, bins=[x_bins, y_bins, z_bins])

            # Project to XY and XZ planes
            hist_xy = np.sum(hist, axis=2)
            hist_xz = np.sum(hist, axis=1)

            # Apply Gaussian smoothing
            hist_xy = gaussian_filter(hist_xy, sigma=sigma)
            hist_xz = gaussian_filter(hist_xz, sigma=sigma)

            # Get subplot indices
            ax_xy, ax_xz = axs[i]

            # Plot XY projection
            ax_xy.imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto', cmap=cmap)
            ax_xy.set_xticks([])  # Remove ticks
            ax_xy.set_yticks([])
            ax_xy.set_title('XY Projection')

            # Plot XZ projection
            ax_xz.imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], aspect='auto', cmap=cmap)
            ax_xz.set_xticks([])
            ax_xz.set_yticks([])
            ax_xz.set_title('XZ Projection')

        # Remove any unused subplots (if number of datasets isn't a multiple of 4)
        for j in range(i + 1, rows * cols):
            axs[j][0].axis("off")
            axs[j][1].axis("off")

        plt.tight_layout()

        # Save or show the grid of plots
        if save_to:
            plt.savefig(save_to, dpi=600)
            print(f"Multi-dataset plots saved to: {save_to}")
            plt.show()
        else:
            plt.show()

        plt.close(fig)  # Free memory

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def plot_multiple_projections_black(data_list, grid_size=0.01, sigma=1, cmap='hot', save_to=None):
    """
    Plot multiple sets of XY and XZ histogram projections with a 3D scatter plot for each sample,
    all on a black background.

    Parameters:
    - data_list (list): A list of datasets (can be file paths, DataFrames, or NumPy arrays).
    - grid_size (float): Size of the grid cells for the histogram bins.
    - sigma (float): Standard deviation for Gaussian smoothing.
    - cmap (str): Colormap for the projections.
    - save_to (str | None): File path to save the plot. If None, the plot is shown instead.
    """
    try:
        num_datasets = len(data_list)
        if num_datasets == 0:
            raise ValueError("The data list must contain at least one dataset.")

        cols = 3  # Columns: [XY Projection | XZ Projection | 3D Scatter]
        rows = len(data_list)

        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), facecolor='black')
        # Adjust spacing: Small space between XY | XZ | 3D, larger space between samples
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.5, hspace=0.8)


        for i, data in enumerate(data_list):
            row_idx = i # Two samples per row
            col_idx = 0  # First column for XY

            # Load data
            if isinstance(data, str):
                data = pd.read_csv(data)
            elif isinstance(data, pd.DataFrame):
                pass
            elif isinstance(data, np.ndarray):
                if data.shape[1] != 3:
                    raise ValueError("Each input array must have 3 columns representing x, y, and z coordinates.")
                data = pd.DataFrame(data, columns=['x', 'y', 'z'])
            else:
                raise ValueError("Each dataset must be a file path, Pandas DataFrame, or NumPy array.")

            # Ensure required columns exist
            if not all(col in data.columns for col in ['x', 'y', 'z']):
                raise ValueError("Each DataFrame must contain the columns: 'x', 'y', and 'z'")

            # Convert DataFrame to NumPy array
            point_cloud = data[['x', 'y', 'z']].to_numpy()

            # Determine histogram bins
            x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
            y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
            z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

            x_bins = int((x_max - x_min) / grid_size) + 1
            y_bins = int((y_max - y_min) / grid_size) + 1
            z_bins = int((z_max - z_min) / grid_size) + 1

            # Compute the 3D histogram
            hist, edges = np.histogramdd(point_cloud, bins=[x_bins, y_bins, z_bins])

            # Project to XY and XZ planes
            hist_xy = np.sum(hist, axis=2)
            hist_xz = np.sum(hist, axis=1)

            # Apply Gaussian smoothing
            hist_xy = gaussian_filter(hist_xy, sigma=sigma)
            hist_xz = gaussian_filter(hist_xz, sigma=sigma)
            # Get the two subplot axes for this dataset
            ax_xy = axs[row_idx, col_idx]  # XY Projection
            ax_xz = axs[row_idx, col_idx + 1]  # XZ Projection
            ax_3d = fig.add_subplot(rows, cols, (row_idx * cols) + col_idx + 3, projection='3d', facecolor='black')

            # Plot XY projection
            ax_xy.imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap=cmap, aspect='auto')
            ax_xy.set_xticks([])
            ax_xy.set_yticks([])
            ax_xy.set_title(f'XY Projection', fontsize=12, color='white')
            ax_xy.set_facecolor('black')

            # Plot XZ projection
            ax_xz.imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], cmap=cmap, aspect='auto')
            ax_xz.set_xticks([])
            ax_xz.set_yticks([])
            ax_xz.set_title(f'XZ Projection', fontsize=12, color='white')
            ax_xz.set_facecolor('black')

            for ax in [ax_xy, ax_xz]:
                ax.spines['top'].set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.spines['right'].set_color('white')
                ax.spines['top'].set_linewidth(1)  # Adjust thickness
                ax.spines['bottom'].set_linewidth(1)
                ax.spines['left'].set_linewidth(1)
                ax.spines['right'].set_linewidth(1)

            # ✅ Center X, Y inside the XY projection
            ax_xy.text(0.5, 0.02, "X", transform=ax_xy.transAxes, fontsize=12, color='white', ha='center', va='bottom')
            ax_xy.text(0.02, 0.5, "Y", transform=ax_xy.transAxes, fontsize=12, color='white', ha='left', va='center',
                       rotation=90)

            # ✅ Center X, Z inside the XZ projection
            ax_xz.text(0.5, 0.02, "X", transform=ax_xz.transAxes, fontsize=12, color='white', ha='center', va='bottom')
            ax_xz.text(0.02, 0.5, "Z", transform=ax_xz.transAxes, fontsize=12, color='white', ha='left', va='center',
                       rotation=90)

            # Plot 3D Scatter (on black)
            ax_3d.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
                          color='skyblue', s=12, alpha=0.6)
            ax_3d.set_title(f'3D Scatter', fontsize=14, color='white')
            # Remove ticks & grid from 3D plot
            ax_3d.set_xticks([])
            ax_3d.set_yticks([])
            ax_3d.set_zticks([])
            ax_3d.set_xlabel("X", fontsize=14, color='white')
            ax_3d.set_ylabel("Y", fontsize=14, color='white')
            ax_3d.set_zlabel("Z", fontsize=14, color='white')

            # Adjust position to move them inside the grid
            ax_3d.xaxis.label.set_position((0.5, -0.3))  # Move X label inside
            ax_3d.yaxis.label.set_position((-0.1, 0.5))  # Move Y label inside
            ax_3d.zaxis.label.set_position((0.5, -0.3))  # Move Z label inside

            ax_3d.grid(False)
            ax_3d.set_facecolor('black')
            #ax_3d.set_frame_on(False)
            ax_3d.patch.set_facecolor('black')
            ax_3d.xaxis.set_pane_color((0, 0, 0, 1))
            ax_3d.yaxis.set_pane_color((0, 0, 0, 1))
            ax_3d.zaxis.set_pane_color((0, 0, 0, 1))
            ax_3d.xaxis.pane.set_edgecolor("white")
            ax_3d.yaxis.pane.set_edgecolor("white")
            ax_3d.zaxis.pane.set_edgecolor("white")

        plt.tight_layout()

        # Save or show the figure
        if save_to:
            plt.savefig(save_to, dpi=600, facecolor='black')
            print(f"Multi-dataset plots saved to: {save_to}")
            plt.show()
        else:
            plt.show()

        plt.close(fig)  # Free memory

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def get_clusters(df, plot=True):
    coordinates = df[['x', 'y', 'z']].values  # Extract coordinates

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)  # Adjust parameters as needed
    df['cluster_id'] = clusterer.fit_predict(coordinates)

    # Filter out noise (HDBSCAN assigns -1 to noise points)
    clustered_df = df[df['cluster_id'] != -1]

    if plot:# Plot results
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(clustered_df['x'], clustered_df['y'], clustered_df['z'], c=clustered_df['cluster_id'], cmap='jet', s=1)
        plt.colorbar(scatter, label="Cluster ID")
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        ax.set_zlabel("Z (nm)")
        plt.title("HDBSCAN Clustering of STORM Localizations")
        plt.show()

    return clustered_df

def get_distance_distribution(clustered_df, plot=True):
    true_positions = clustered_df.groupby('cluster_id')[['x', 'y', 'z']].median().reset_index()
    df = clustered_df.merge(true_positions, on="cluster_id", suffixes=("", "_true"))
    df["radial_distance"] = np.linalg.norm(df[["x", "y", "z"]].values - df[["x_true", "y_true", "z_true"]].values, axis=1)
    if plot:
        plt.figure(figsize=(8,6))
        sns.histplot(df["radial_distance"], bins=50, kde=True)
        plt.xlabel("Radial Distance from True Position (nm)")
        plt.ylabel("Frequency")
        plt.title("Experimental Blink Spread")
        plt.show()
    return df

def fit_normal_expo_distribution(df):
    # Fit Gaussian distribution
    mu, std = np.mean(df["radial_distance"]), np.std(df["radial_distance"])
    gaussian_pdf = norm.pdf(np.linspace(0, np.max(df["radial_distance"]), 100), loc=mu, scale=std)
    # Fit Exponential distribution
    lambda_exp = 1 / np.mean(df["radial_distance"])
    exp_pdf = lambda_exp * np.exp(-lambda_exp * np.linspace(0, np.max(df["radial_distance"]), 100))
    # Plot Histogram with Gaussian & Exponential Fits
    plt.figure(figsize=(8,6))
    sns.histplot(df["radial_distance"], bins=50, kde=False, label="Experimental Data")
    plt.plot(np.linspace(0, np.max(df["radial_distance"]), 100), gaussian_pdf, label="Gaussian Fit", linewidth=2)
    plt.plot(np.linspace(0, np.max(df["radial_distance"]), 100), exp_pdf, label="Exponential Fit", linewidth=2)
    plt.xlabel("Radial Distance from True Position (nm)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Comparison of Localization Precision Models")
    plt.show()

def fit_rayleigh_normal_distribution(df):
    # Compute best-fit Rayleigh distribution
    sigma_fit = np.sqrt(np.mean(df["radial_distance"] ** 2) / 2)  # Estimate sigma from data
    rayleigh_pdf = rayleigh.pdf(np.linspace(0, np.max(df["radial_distance"]), 100), scale=sigma_fit)

    # Fit a Gaussian to compare
    mu, std = np.mean(df["radial_distance"]), np.std(df["radial_distance"])
    gaussian_pdf = norm.pdf(np.linspace(0, np.max(df["radial_distance"]), 100), loc=mu, scale=std)

    # Plot histogram with Rayleigh and Gaussian fits
    plt.figure(figsize=(8, 6))
    sns.histplot(df["radial_distance"], bins=50, kde=False, label="Experimental Data")
    plt.plot(np.linspace(0, np.max(df["radial_distance"]), 100), rayleigh_pdf, label="Rayleigh Fit", linewidth=2)
    plt.plot(np.linspace(0, np.max(df["radial_distance"]), 100), gaussian_pdf, label="Gaussian Fit", linewidth=2)
    plt.xlabel("Radial Distance from True Position (nm)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Comparison of Localization Precision Models (Rayleigh vs Gaussian)")
    plt.show()

def fit_maxwell_normal_distribution(df):
    # Compute best-fit Maxwell-Boltzmann distribution
    sigma_fit = np.sqrt(np.mean(df["radial_distance"] ** 2) / 3)  # 3D equivalent of Rayleigh
    maxwell_pdf = maxwell.pdf(np.linspace(0, np.max(df["radial_distance"]), 100), scale=sigma_fit)

    # Fit a Gaussian to compare
    mu, std = np.mean(df["radial_distance"]), np.std(df["radial_distance"])
    gaussian_pdf = np.exp(-0.5 * ((np.linspace(0, np.max(df["radial_distance"]), 100) - mu) / std) ** 2)

    # Plot histogram with Maxwell-Boltzmann and Gaussian fits
    plt.figure(figsize=(8, 6))
    sns.histplot(df["radial_distance"], bins=50, kde=False, label="Experimental Data")
    plt.plot(np.linspace(0, np.max(df["radial_distance"]), 100), maxwell_pdf, label="Maxwell-Boltzmann Fit",
             linewidth=2)
    plt.plot(np.linspace(0, np.max(df["radial_distance"]), 100), gaussian_pdf, label="Gaussian Fit", linewidth=2)
    plt.xlabel("Radial Distance from True Position (nm)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Comparison of Localization Precision Models (Maxwell-Boltzmann vs Gaussian)")
    plt.show()


def fit_multiple_maxwell_normal_distribution(df1, df2, labels=("Dataset 1", "Dataset 2")):
    """Fit and plot Maxwell-Boltzmann and Gaussian distributions for two datasets."""

    def compute_distributions(df):
        """Compute Maxwell-Boltzmann and Gaussian distributions for a given dataframe."""
        sigma_fit = np.sqrt(np.mean(df["radial_distance"] ** 2) / 3)  # 3D equivalent of Rayleigh
        maxwell_pdf = maxwell.pdf(np.linspace(0, np.max(df["radial_distance"]), 100), scale=sigma_fit)

        mu, std = np.mean(df["radial_distance"]), np.std(df["radial_distance"])
        gaussian_pdf = np.exp(-0.5 * ((np.linspace(0, np.max(df["radial_distance"]), 100) - mu) / std) ** 2)

        return maxwell_pdf, gaussian_pdf

    # Compute distributions for both datasets
    maxwell_pdf1, gaussian_pdf1 = compute_distributions(df1)
    maxwell_pdf2, gaussian_pdf2 = compute_distributions(df2)

    # Plot histogram and fits
    plt.figure(figsize=(8, 6))

    # Histogram for both datasets
    sns.histplot(df1["radial_distance"], bins=50, kde=False, label=f"{labels[0]} Data", alpha=0.5)
    sns.histplot(df2["radial_distance"], bins=50, kde=False, label=f"{labels[1]} Data", alpha=0.5)

    # X-axis range
    x_range = np.linspace(0, max(np.max(df1["radial_distance"]), np.max(df2["radial_distance"])), 100)

    # Plot Maxwell-Boltzmann fits
    plt.plot(x_range, maxwell_pdf1, label=f"{labels[0]} Maxwell-Boltzmann Fit", linewidth=2)
    plt.plot(x_range, maxwell_pdf2, label=f"{labels[1]} Maxwell-Boltzmann Fit", linewidth=2, linestyle="dashed")

    # Plot Gaussian fits
    plt.plot(x_range, gaussian_pdf1, label=f"{labels[0]} Gaussian Fit", linewidth=2)
    plt.plot(x_range, gaussian_pdf2, label=f"{labels[1]} Gaussian Fit", linewidth=2, linestyle="dashed")

    # Labels and legend
    plt.xlabel("Radial Distance from True Position (nm)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Comparison of Localization Precision Models (Maxwell-Boltzmann vs Gaussian)")

    # Show plot
    plt.show()

def calculate_distance_distributions(df):
    clustered_df = get_clusters(df)
    distrib = get_distance_distribution(clustered_df)
    return distrib

def fit_maxwell_rayleigh_normal(df):
    # Compute best-fit Rayleigh distribution
    sigma_fit = np.sqrt(np.mean(df["radial_distance"] ** 2) / 2)  # Estimate sigma from data
    rayleigh_pdf = rayleigh.pdf(np.linspace(0, np.max(df["radial_distance"]), 100), scale=sigma_fit)

    # Compute best-fit Maxwell-Boltzmann distribution
    sigma_fit = np.sqrt(np.mean(df["radial_distance"] ** 2) / 3)  # 3D equivalent of Rayleigh
    maxwell_pdf = maxwell.pdf(np.linspace(0, np.max(df["radial_distance"]), 100), scale=sigma_fit)

    # Fit a Gaussian to compare
    mu, std = np.mean(df["radial_distance"]), np.std(df["radial_distance"])
    gaussian_pdf = np.exp(-0.5 * ((np.linspace(0, np.max(df["radial_distance"]), 100) - mu) / std) ** 2)

    # Plot histogram with Maxwell-Boltzmann and Gaussian fits
    plt.figure(figsize=(8, 6))
    sns.histplot(df["radial_distance"], bins=50, kde=False, label="Experimental Data")
    plt.plot(np.linspace(0, np.max(df["radial_distance"]), 100), maxwell_pdf, label="Maxwell-Boltzmann Fit",
             linewidth=2)
    plt.plot(np.linspace(0, np.max(df["radial_distance"]), 100), gaussian_pdf, label="Gaussian Fit", linewidth=2)
    plt.plot(np.linspace(0, np.max(df["radial_distance"]), 100), rayleigh_pdf, label="Rayleigh Fit", linewidth=2)
    plt.xlabel("Radial Distance from True Position (nm)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.title("Comparison of Localization Precision Models (Maxwell-Boltzmann vs Gaussian)")
    plt.show()

def statistical_test(df1, df2):
    from scipy.stats import ks_2samp, mannwhitneyu
    ks_stat, ks_pval = ks_2samp(df1["radial_distance"], df2["radial_distance"])
    u_stat, u_pval = mannwhitneyu(df1["radial_distance"], df2["radial_distance"])

    print(f"KS Test p-value: {ks_pval:.4f} (Shape Difference)")
    print(f"Mann-Whitney U Test p-value: {u_pval:.4f} (Median Shift)")
    return ks_stat, ks_pval

# Compute estimated localization precision from Maxwell-Boltzmann fit (?)
#sigma_fit = np.sqrt(np.mean(distrib["radial_distance"]**2) / 3)  # 3D localization precision


# Define the Thompson-Mortensen precision formula
def thompson_mortensen_precision(photons, psf_size, pixel_size, background):
    return np.sqrt((psf_size ** 2 + pixel_size ** 2 / 12) / photons +
                   (8 * np.pi * psf_size ** 4 * background) / (photons ** 2 * pixel_size ** 2))

# ? Define your experimental parameters (Update with actual values)
psf_size = 30  # nm (Point Spread Function size)
pixel_size = 100  # nm (Camera pixel size)
background = 10  # photons per pixel
mean_photons = 5000  # Mean detected photons per localization

# Compute theoretical localization precision
predicted_precision = thompson_mortensen_precision(mean_photons, psf_size, pixel_size, background)

import glob
csv_files = glob.glob(r"C:\Users\diana\PhD\data\tetra_heydarian\tetra\*.csv")  # Adjust this path

sigma_values = []

for file in csv_files:
    # Load dataset
    df = pd.read_csv(file)
    df = get_clusters(df, plot=False)

    df = get_distance_distribution(df, plot=False)

    # Compute sigma (localization spread) using Maxwell-Boltzmann fit
    sigma_fit = np.sqrt(np.mean(df["radial_distance"] ** 2) / 3)

    # Store sigma value for this sample
    sigma_values.append(sigma_fit)

# Compute the overall average ? across all samples
average_sigma = np.mean(sigma_values)
#std_sigma = np.std(sigma_values) 

# Extract different individual stats from big stats dict

from helpers.pc_stats import load_and_analyze_point_clouds
pathh = '/home/dim26fa/data/sim_25022025_5'
stats = load_and_analyze_point_clouds(pathh)


#stats = stats_exp
num_points = [entry['num_points'] for item in stats for key, entry in item.items() if 'num_points' in entry]
cluster_sizes = [entry['cluster_size'] for item in stats for key, entry in item.items() if 'cluster_size' in entry]
cluster_stds = [entry['cluster_std'] for item in stats for key, entry in item.items() if 'cluster_std' in entry]
sample_stds = [entry['sample_std'] for item in stats for key, entry in item.items() if 'sample_std' in entry]  # Assuming 'var' is variance



# Extract standard deviation values
cluster_std_x, cluster_std_y, cluster_std_z = [], [], []
sample_std_x, sample_std_y, sample_std_z = [], [], []

for series in cluster_stds:
    if isinstance(series, pd.Series):
        cluster_std_x.append(series['x'])
        cluster_std_y.append(series['y'])
        cluster_std_z.append(series['z'])

for series in sample_stds:
    if isinstance(series, pd.Series):
        sample_std_x.append(series['x'])
        sample_std_y.append(series['y'])
        sample_std_z.append(series['z'])
fig, axes = plt.subplots(2, 3, figsize=(7.68, 5.12))
# First row: Cluster Std (X, Y, Z) - Histograms
sns.histplot(cluster_std_x, kde=True, bins=30, color='steelblue', edgecolor='black', alpha=0.8, ax=axes[0, 0])
axes[0, 0].set_title("Cluster Std (X)", fontsize=10)
axes[0, 0].axvline(np.mean(cluster_std_x), color='red', linestyle='dashed', label='Mean')
sns.histplot(cluster_std_y, kde=True, bins=30, color='steelblue', edgecolor='black', alpha=0.8, ax=axes[0, 1])
axes[0, 1].set_title("Cluster Std (Y)", fontsize=10)
axes[0, 1].axvline(np.mean(cluster_std_y), color='red', linestyle='dashed', label='Mean')
sns.histplot(cluster_std_z, kde=True, bins=30, color='steelblue', edgecolor='black', alpha=0.8, ax=axes[0, 2])
axes[0, 2].set_title("Cluster Std (Z)", fontsize=10)
axes[0, 2].axvline(np.mean(cluster_std_z), color='red', linestyle='dashed', label='Mean')

# Second row: Sample Std (X, Y, Z) - Histograms
sns.histplot(sample_std_x, kde=True, bins=30, color='steelblue', edgecolor='black', alpha=0.8, ax=axes[1, 0])
axes[1, 0].set_title("Sample Std (X)", fontsize=10)
axes[1, 0].axvline(np.mean(sample_std_x), color='red', linestyle='dashed', label='Mean')
sns.histplot(sample_std_y, kde=True, bins=30, color='steelblue', edgecolor='black', alpha=0.8, ax=axes[1, 1])
axes[1, 1].set_title("Sample Std (Y)", fontsize=10)
axes[1, 1].axvline(np.mean(sample_std_y), color='red', linestyle='dashed', label='Mean')

sns.histplot(sample_std_z, kde=True, bins=30, color='steelblue', edgecolor='black', alpha=0.8, ax=axes[1, 2])
axes[1, 2].set_title("Sample Std (Z)", fontsize=10)
axes[1, 2].axvline(np.mean(sample_std_z), color='red', linestyle='dashed', label='Mean')

plt.tight_layout()
#plt.savefig('/home/dim26fa/figures/sim_stats_std2.png', dpi=600)
plt.show()

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(7.68, 5.12))

# Histogram with density plot for num_points
sns.histplot(num_points, kde=True, bins=50, ax=axes[0, 0], color="teal")
axes[0, 0].axvline(np.mean(num_points), color='red', linestyle='dashed', label='Mean')
axes[0, 0].set_title("Number of Points")
axes[0, 0].set_ylabel("Count")

# Histogram with density plot for cluster_sizes
sns.histplot(cluster_sizes, kde=True, bins=50, ax=axes[0, 1], color="teal")
axes[0, 1].axvline(np.mean(cluster_sizes), color='red', linestyle='dashed', label='Mean')
axes[0, 1].set_title("Cluster Sizes")
axes[0, 1].set_ylabel("Count")

# Boxplot for num_points
sns.boxplot(y=num_points, ax=axes[1, 0], color="teal")
axes[1, 0].set_title("Number of Points")

# Boxplot for cluster_sizes
sns.boxplot(y=cluster_sizes, ax=axes[1, 1], color="teal")
axes[1, 1].set_title("Cluster Sizes")

# Adjust layout
plt.tight_layout()
#plt.savefig('/home/dim26fa/figures/sim_stats_num_points2.png', dpi=600)
plt.show()