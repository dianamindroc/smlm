import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
import seaborn as sns
from scipy.stats import rayleigh, norm, maxwell
from scipy.ndimage import gaussian_filter
from matplotlib_scalebar.scalebar import ScaleBar
from sklearn.manifold import TSNE
from scipy.signal import find_peaks
import os
from matplotlib.colors import LogNorm


def plot_xy_xz_histograms(data, grid_size=0.01, cmap='hot', save_to=None, units='nm', dx=90, z_cutoff=None, percentile_vmax=99.5, use_log_norm=False):
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

        # Filter outliers based on Z values if z_cutoff is provided
        if z_cutoff is not None:
            if isinstance(z_cutoff, tuple) and len(z_cutoff) == 2:
                # Use explicit min/max values
                z_min_cutoff, z_max_cutoff = z_cutoff
                filtered_point_cloud = point_cloud[
                    (point_cloud[:, 2] >= z_min_cutoff) &
                    (point_cloud[:, 2] <= z_max_cutoff)
                ]
            elif isinstance(z_cutoff, (int, float)):
                # Use percentile-based filtering
                z_values = point_cloud[:, 2]
                lower_bound = np.percentile(z_values, (100 - z_cutoff) / 2)
                upper_bound = np.percentile(z_values, 100 - (100 - z_cutoff) / 2)
                filtered_point_cloud = point_cloud[
                    (point_cloud[:, 2] >= lower_bound) &
                    (point_cloud[:, 2] <= upper_bound)
                ]
            else:
                filtered_point_cloud = point_cloud
        else:
            filtered_point_cloud = point_cloud

        point_cloud = filtered_point_cloud
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

        # --- Calculate Colormap Limits (vmax or LogNorm) ---
        norm_xy, norm_xz = None, None
        vmax_xy, vmax_xz = None, None

        if use_log_norm:
            # Use LogNorm - find min non-zero value for vmin robustly
            min_xy = np.min(hist_xy[hist_xy > 1e-9]) if np.any(hist_xy > 1e-9) else 1e-9
            min_xz = np.min(hist_xz[hist_xz > 1e-9]) if np.any(hist_xz > 1e-9) else 1e-9
            norm_xy = LogNorm(vmin=min_xy, vmax=hist_xy.max())
            norm_xz = LogNorm(vmin=min_xz, vmax=hist_xz.max())
        else:
            # Use percentile clipping for vmax
            non_zero_xy = hist_xy[hist_xy > 1e-9]
            non_zero_xz = hist_xz[hist_xz > 1e-9]
            vmax_xy = np.percentile(non_zero_xy, percentile_vmax) if non_zero_xy.size > 0 else 1
            vmax_xz = np.percentile(non_zero_xz, percentile_vmax) if non_zero_xz.size > 0 else 1


        # Visualize the XY and XZ projections
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # XY Projection
        axs[0].imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto', cmap=cmap, norm=norm_xy, vmax=vmax_xy)
        axs[0].set_title('XY Projection')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')
        #axs[0].set_xticks([])  # Remove ticks
        #axs[0].set_yticks([])

        # Add scale bar to XY projection
        scalebar_xy = ScaleBar(
            dx= dx,  # 1.0 means coordinates are in real-world units
            units=units,
            location='lower right',
            frameon=True,
            color='white',
            box_alpha=0.5,
            box_color='black'
        )
        axs[0].add_artist(scalebar_xy)


        # XZ Projection
        axs[1].imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], aspect='auto', cmap=cmap, norm=norm_xz, vmax=vmax_xz)
        axs[1].set_title('XZ Projection')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Z')
        #axs[1].set_xticks([])  # Remove ticks
        #axs[1].set_yticks([])
        # Add scale bar to XZ projection
        scalebar_xz = ScaleBar(
            dx=dx,  # 1.0 means coordinates are in real-world units
            units=units,
            location='lower right',
            frameon=True,
            color='white',
            box_alpha=0.5,
            box_color='black'
        )
        axs[1].add_artist(scalebar_xz)

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


def plot_xy_xz_histograms_with_hdbscan(data, grid_size=0.01, min_cluster_size=10, min_samples=5,
                                       cluster_selection_epsilon=0.0, cluster_selection_method='eom',
                                       cmap='viridis', noise_color='gray', save_to=None):
    """
    Performs HDBSCAN clustering on 3D point cloud data and plots XY and XZ projections.

    Parameters:
    - data (str | pd.DataFrame | np.ndarray): Can be a file path, DataFrame, or NumPy array.
    - grid_size (float): Size of the grid cells for the histogram bins.
    - min_cluster_size (int): Minimum size of clusters (HDBSCAN parameter).
    - min_samples (int): Min samples param for HDBSCAN core point detection.
    - cluster_selection_epsilon (float): Radius for cluster merging in HDBSCAN.
    - cluster_selection_method (str): Method for cluster extraction ('eom' or 'leaf').
    - cmap (str): Color map for the clusters.
    - noise_color (str): Color for noise points (label -1 in HDBSCAN).
    - save_to (str | None): File path to save the plot.

    Returns:
    - labels (np.ndarray): HDBSCAN cluster labels for each point
    - clusterer (hdbscan.HDBSCAN): HDBSCAN clustering model
    """
    try:
        import hdbscan
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter

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

        # Extract point cloud
        point_cloud = data[['x', 'y', 'z']].to_numpy()

        # Perform HDBSCAN clustering
        print("Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method
        )
        labels = clusterer.fit_predict(point_cloud)

        # Add cluster labels to DataFrame
        data['cluster'] = labels

        # Print clustering statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise} ({n_noise / len(labels):.2%} of total)")

        # Determine the range for plots
        x_min, x_max = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
        y_min, y_max = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
        z_min, z_max = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])

        # Create the figure with subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Get unique cluster labels (excluding noise if present)
        unique_clusters = np.unique(labels)
        is_noise_present = -1 in unique_clusters

        # Create a color map for the clusters
        if is_noise_present:
            # Remove noise label (-1) from color mapping
            cluster_labels = unique_clusters[unique_clusters != -1]
            n_clusters = len(cluster_labels)
            colors = plt.cm.get_cmap(cmap, n_clusters)
        else:
            n_clusters = len(unique_clusters)
            colors = plt.cm.get_cmap(cmap, n_clusters)

        # Plot XY Projection
        axs[0].set_title('XY Projection with HDBSCAN Clusters')
        axs[0].set_xlabel('X')
        axs[0].set_ylabel('Y')

        # Plot XZ Projection
        axs[1].set_title('XZ Projection with HDBSCAN Clusters')
        axs[1].set_xlabel('X')
        axs[1].set_ylabel('Z')

        # Plot each cluster
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:
                # Plot noise points
                noise_points = data[data['cluster'] == -1]
                axs[0].scatter(noise_points['x'], noise_points['y'],
                               c=noise_color, marker='.', alpha=0.5, s=20, label='Noise')
                axs[1].scatter(noise_points['x'], noise_points['z'],
                               c=noise_color, marker='.', alpha=0.5, s=20, label='Noise')
            else:
                # Plot cluster points
                cluster_points = data[data['cluster'] == cluster_id]
                color_idx = i if not is_noise_present else i - 1
                color = colors(color_idx % n_clusters)  # Cycle through colors if needed

                axs[0].scatter(cluster_points['x'], cluster_points['y'],
                               c=[color], marker='.', alpha=0.8, s=20,
                               label=f'Cluster {cluster_id}')

                axs[1].scatter(cluster_points['x'], cluster_points['z'],
                               c=[color], marker='.', alpha=0.8, s=20)

        # Add legends (may need to be adjusted for many clusters)
        if n_clusters <= 10:  # Only add legend if not too many clusters
            axs[0].legend(loc='best', markerscale=2)

        plt.tight_layout()

        # Save or show the plot
        if save_to:
            plt.savefig(save_to, dpi=600)
            print(f"Plot saved to: {save_to}")
        else:
            plt.show()

        plt.close(fig)  # Close the figure after saving or displaying it


    except ImportError:
        print("Error: hdbscan module not found. Install with 'pip install hdbscan'.")
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


def plot_multiple_projections(data_list, grid_size=0.01, sigma=1, cmap='hot', save_to=None, units='nm', dx=100, z_cutoff=None):
    """
    Plot multiple sets of XY and XZ histogram projections, stacking the plots in a grid with scale bars.

    Parameters:
    - data_list (list): A list of datasets (can be file paths, DataFrames, or NumPy arrays).
    - grid_size (float): Size of the grid cells for the histogram bins.
    - sigma (float): Sigma for Gaussian smoothing.
    - cmap (str): Colormap to use for visualization.
    - save_to (str | None): File path to save the plot. If None, the plot is shown instead.
    - units (str): Units for the scale bar (e.g., 'nm', 'µm').
    - dx (float): Scale factor for the scale bar (physical size of one data unit).
    """
    try:
        # Import required library for scale bars
        from matplotlib_scalebar.scalebar import ScaleBar

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

            # Filter outliers based on Z values if z_cutoff is provided
            if z_cutoff is not None:
                if isinstance(z_cutoff, tuple) and len(z_cutoff) == 2:
                    # Use explicit min/max values
                    z_min_cutoff, z_max_cutoff = z_cutoff
                    filtered_point_cloud = point_cloud[
                        (point_cloud[:, 2] >= z_min_cutoff) &
                        (point_cloud[:, 2] <= z_max_cutoff)
                        ]
                elif isinstance(z_cutoff, (int, float)):
                    # Use percentile-based filtering
                    z_values = point_cloud[:, 2]
                    lower_bound = np.percentile(z_values, (100 - z_cutoff) / 2)
                    upper_bound = np.percentile(z_values, 100 - (100 - z_cutoff) / 2)
                    filtered_point_cloud = point_cloud[
                        (point_cloud[:, 2] >= lower_bound) &
                        (point_cloud[:, 2] <= upper_bound)
                        ]
                else:
                    filtered_point_cloud = point_cloud
            else:
                filtered_point_cloud = point_cloud

            point_cloud = filtered_point_cloud

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

            # Add scale bar to XY projection
            scalebar_xy = ScaleBar(
                dx=dx,
                units=units,
                length_fraction=0.15,
                height_fraction=0.02,
                location='lower right',
                frameon=False,
                color='white',
                box_alpha=0,
                scale_loc='bottom'
            )
            axs[i][0].add_artist(scalebar_xy)
            axs[i][0].set_xticks([])
            axs[i][0].set_yticks([])

            # Plot the XZ projection
            axs[i][1].imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], aspect='auto', cmap=cmap)
            axs[i][1].set_title('XZ Projection')
            axs[i][1].set_xlabel('X')
            axs[i][1].set_ylabel('Z')

            # Add scale bar to XZ projection with the same settings for consistency
            scalebar_xz = ScaleBar(
                dx=dx,
                units=units,
                length_fraction=0.15,
                height_fraction=0.02,
                location='lower right',
                frameon=False,
                color='white',
                box_alpha=0,
                scale_loc='bottom'
            )
            axs[i][1].add_artist(scalebar_xz)
            axs[i][1].set_xticks([])
            axs[i][1].set_yticks([])

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


def plot_multiple_projections_ordered(data_list, grid_size=0.01, sigma=1, cmap='hot', cols=2, save_to=None, column_titles=None, dx=130, units='nm'):
    """
       Plot multiple sets of XY and XZ histogram projections in a grid.
       Datasets are arranged in columns (first 'rows_per_col' datasets in column 1, next 'rows_per_col' in column 2, etc.)

       Parameters:
       - data_list (list): A list of datasets (can be file paths, DataFrames, or NumPy arrays).
       - grid_size (float): Size of the grid cells for the histogram bins.
       - sigma (float): Standard deviation for Gaussian smoothing.
       - cmap (str): Colormap for the projections.
       - cols (int): Number of columns to arrange datasets (default 2).
       - save_to (str | None): File path to save the plot. If None, the plot is shown instead.
       - column_titles (list | None): List of titles for each column. Defaults to ["GT", "Output"] for 2 columns.
       """
    try:
        num_datasets = len(data_list)
        if num_datasets == 0:
            raise ValueError("The data list must contain at least one dataset.")

        # Set default column titles if not provided
        if column_titles is None:
            column_titles = ["GT", "Output"] + [f"Column {i + 3}" for i in range(cols - 2)]

        # Ensure we have enough titles for all columns
        if len(column_titles) < cols:
            column_titles.extend([f"Column {i + 1 + len(column_titles)}" for i in range(cols - len(column_titles))])

        # Calculate number of rows needed per column
        rows_per_col = (num_datasets + cols - 1) // cols

        # Create a figure with the appropriate grid
        # Each dataset needs 2 subplots (XY and XZ), and we arrange datasets in columns
        fig, axs = plt.subplots(rows_per_col, cols * 2, figsize=(6 * cols, 4 * rows_per_col))

        # Ensure axs is always a 2D array
        if rows_per_col == 1 and cols == 1:
            axs = axs.reshape(1, 2)
        elif rows_per_col == 1:
            axs = axs.reshape(1, -1)
        elif cols == 1:
            axs = axs.reshape(-1, 2)

        # Process each dataset
        for i, data in enumerate(data_list):
            # Determine which column and row within that column this dataset belongs to
            col_idx = i // rows_per_col
            row_idx = i % rows_per_col

            # Skip if we've run out of columns
            if col_idx >= cols:
                print(f"Warning: Dataset {i + 1} exceeds the specified grid. Consider increasing 'cols'.")
                continue

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

            # Calculate subplot indices
            # Each column gets 2 subplots (XY and XZ) side by side
            xy_idx = col_idx * 2
            xz_idx = col_idx * 2 + 1

            # Get the right axes objects
            if rows_per_col == 1:
                ax_xy = axs[xy_idx]
                ax_xz = axs[xz_idx]
            else:
                ax_xy = axs[row_idx, xy_idx]
                ax_xz = axs[row_idx, xz_idx]

            # Plot XY projection
            ax_xy.imshow(hist_xy.T, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto', cmap=cmap)
            ax_xy.set_xticks([])  # Remove ticks
            ax_xy.set_yticks([])

            # Set title with column label from column_titles
            col_label = column_titles[col_idx] if col_idx < len(column_titles) else f"Column {col_idx + 1}"
            ax_xy.set_title(f'{col_label}: XY')

            # Plot XZ projection
            ax_xz.imshow(hist_xz.T, origin='lower', extent=[x_min, x_max, z_min, z_max], aspect='auto', cmap=cmap)
            ax_xz.set_xticks([])
            ax_xz.set_yticks([])

            # Set title with column label from column_titles
            col_label = column_titles[col_idx] if col_idx < len(column_titles) else f"Column {col_idx + 1}"
            ax_xz.set_title(f'{col_label}: XZ')

            # Add scale bar to XY projection
            scalebar_xy = ScaleBar(
                dx=dx,  # 1.0 means coordinates are in real-world units
                units=units,
                location='lower right',
                frameon=True,
                color='white',
                box_alpha=0.5,
                box_color='black'
            )
            ax_xy.add_artist(scalebar_xy)

            # Add scale bar to XZ projection
            scalebar_xz = ScaleBar(
                dx=dx,  # 1.0 means coordinates are in real-world units
                units=units,
                location='lower right',
                frameon=True,
                color='white',
                box_alpha=0.5,
                box_color='black'
            )
            ax_xz.add_artist(scalebar_xz)


        # Turn off any unused subplots
        for i in range(num_datasets, rows_per_col * cols):
            col_idx = i // rows_per_col
            row_idx = i % rows_per_col
            xy_idx = col_idx * 2
            xz_idx = col_idx * 2 + 1

            if rows_per_col == 1:
                if xy_idx < axs.shape[0]:
                    axs[xy_idx].axis('off')
                if xz_idx < axs.shape[0]:
                    axs[xz_idx].axis('off')
            else:
                if col_idx < cols:
                    axs[row_idx, xy_idx].axis('off')
                    axs[row_idx, xz_idx].axis('off')

        plt.tight_layout()

        # Save or show the grid of plots
        if save_to:
            plt.savefig(save_to, dpi=600)
            print(f"Multi-dataset plots saved to: {save_to}")
        else:
            plt.show()

        plt.close(fig)  # Free memory

    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def plot_multiple_projections_black(data_list, grid_size=0.01, sigma=1, cmap='hot', dx=100, units='nm', save_to=None):
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
            point_cloud = remove_outliers(point_cloud)

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

            scalebar_xy = ScaleBar(
                dx=dx,  # Set this to your correct scale factor (e.g., 100 nm per unit)
                units=units,
                length_fraction=0.15,
                height_fraction=0.02,
                location='lower right',
                frameon=False,
                color='white',
                box_alpha=0,
                scale_loc='bottom'
            )
            ax_xy.add_artist(scalebar_xy)

            # Add scale bar to XZ projection with same settings
            scalebar_xz = ScaleBar(
                dx=dx,  # Use the same scale factor for consistency
                units=units,
                length_fraction=0.15,
                height_fraction=0.02,
                location='lower right',
                frameon=False,
                color='white',
                box_alpha=0,
                scale_loc='bottom'
            )
            ax_xz.add_artist(scalebar_xz)
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


def cluster_and_plot(features, min_cluster_size=5, tsne_perplexity=30, random_state=42, save_to=None):
    """
    Clusters feature vectors using HDBSCAN and plots them in 2D using t-SNE.

    Parameters:
    - features: List or np.ndarray of shape (n_samples, n_features)
    - min_cluster_size: Minimum size of clusters for HDBSCAN
    - tsne_perplexity: Perplexity parameter for t-SNE
    - random_state: Random seed for reproducibility
    """
    features = np.array(features)

    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(features)

    # Run t-SNE for visualization
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=random_state)
    features_2d = tsne.fit_transform(features)

    # Plot t-SNE colored by cluster label
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label != -1:  # Skip noise points
            mask = labels == label
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], label=f'Cluster {label}', alpha=0.6, s=50)

    plt.title('t-SNE Visualization of HDBSCAN Clusters')
    plt.legend()
    plt.grid(False)
    plt.tight_layout()

    # Save or show the grid of plots
    if save_to:
        plt.savefig(save_to, dpi=600)
        print(f"Multi-dataset plots saved to: {save_to}")
        plt.show()
    else:
        plt.show()

    plt.close('all')  # Free memory

    return labels


def analyze_ring_radii(points, nbins=50, z_bins=40, nm_per_unit=1.0,
                       z_peak_height_fraction=0.3, r_peak_height_fraction=0.3,
                       plot=True, save_to=None):
    """
    Analyzes a 3D point cloud (presumably of an NPC) to estimate the radii
    of the upper and lower rings.

    It splits the point cloud along the z-axis based on density peaks and then
    calculates the radius for each half using a radial histogram centered
    on the xy-centroid of that half.

    Args:
        points (np.ndarray): Input point cloud data, shape (N, 3).
        nbins (int): Number of bins for the radial histograms. Defaults to 50.
        z_bins (int): Number of bins for the z-axis histogram. Defaults to 40.
        nm_per_unit (float): Scaling factor to convert input point units to nm.
                              Defaults to 1.0 (assumes input is already in nm).
        z_peak_height_fraction (float): Relative height threshold (fraction of max)
                                       for finding peaks in the z-histogram.
                                       Defaults to 0.3.
        r_peak_height_fraction (float): Relative height threshold (fraction of max)
                                       for finding peaks in the radial histograms.
                                       Defaults to 0.3.
        plot (bool): If True, generates plots of the z-histogram and radial
                     histograms. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - radius_lower (float or None): Estimated radius of the lower ring in nm.
                                            None if calculation failed.
            - radius_upper (float or None): Estimated radius of the upper ring in nm.
                                            None if calculation failed.
            - center_lower (tuple or None): (x, y) coordinates of the lower ring's
                                            centroid in nm. None if calculation failed.
            - center_upper (tuple or None): (x, y) coordinates of the upper ring's
                                            centroid in nm. None if calculation failed.
            - z_split (float or None): The z-coordinate used to split the data in nm.
                                        None if splitting failed.
    """
    if points.shape[0] < 10: # Need some points to work with
         print("Warning: Very few points in input data.")
         return None, None, None, None, None

    # Scale the point cloud
    points_scaled = points * nm_per_unit
    x, y, z = points_scaled[:, 0], points_scaled[:, 1], points_scaled[:, 2]

    # --- Z-axis analysis and splitting ---
    z_split = None
    try:
        hist_z, bin_edges_z = np.histogram(z, bins=z_bins)
        centers_z = (bin_edges_z[:-1] + bin_edges_z[1:]) / 2

        # Find peaks (potential rings) along z
        # Avoid division by zero if hist_z is empty or flat
        max_hist_z = np.max(hist_z) if hist_z.size > 0 else 0
        if max_hist_z > 0:
             peaks_z, _ = find_peaks(hist_z, height=max_hist_z * z_peak_height_fraction)
        else:
             peaks_z = np.array([]) # No peaks if histogram is flat/empty

        split_method = "N/A"
        if len(peaks_z) >= 2:
            # If two or more peaks found, assume they are the rings. Find valley between the first two.
            peak_indices_sorted = sorted(peaks_z) # Sort by index (which corresponds to z)
            first_peak_idx = peak_indices_sorted[0]
            second_peak_idx = peak_indices_sorted[1]

            # Ensure indices are valid for slicing
            if first_peak_idx < second_peak_idx:
                # Find the minimum point (valley) in the histogram between the first two peaks
                valley_bin_index = np.argmin(hist_z[first_peak_idx:second_peak_idx+1]) + first_peak_idx
                z_split = centers_z[valley_bin_index]
                split_method = "Valley between Z-peaks"
            else: # Should not happen if sorted, but as safety
                 print("Warning: Z-peak indices calculation error. Falling back to median.")
                 peaks_z = [] # Force fallback

        # Fallback or if only 0 or 1 peak found
        if z_split is None:
            z_split = np.median(z)
            split_method = "Median Z (fallback)"
            if len(peaks_z) < 2:
                 print(f"Warning: Found {len(peaks_z)} peak(s) in Z-histogram. Using median Z ({z_split:.2f} nm) for splitting.")

    except Exception as e:
        print(f"Error during Z-axis analysis: {e}. Cannot proceed with splitting.")
        return None, None, None, None, None

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(centers_z, hist_z, label='Z-distribution')
        if len(peaks_z) > 0:
            plt.plot(centers_z[peaks_z], hist_z[peaks_z], "x", markersize=10, label='Detected Z-peaks')
        if z_split is not None:
            plt.axvline(z_split, color='r', linestyle='--', label=f'Split Plane ({split_method}): {z_split:.2f} nm')
        plt.title('Z-axis Histogram and Split Plane')
        plt.xlabel('Z-coordinate (nm)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Split data based on calculated z_split
    lower_half = points_scaled[z <= z_split]
    upper_half = points_scaled[z > z_split]

    # --- Radial analysis for each half ---
    def get_radial_hist_and_peak(group, label, nbins, r_peak_height_fraction, plot, save_to=None):
        """Inner function to calculate radius and center for a point group."""
        if group.shape[0] < 5: # Need a minimum number of points for meaningful centroid/radius
            print(f"Warning: Not enough points ({group.shape[0]}) in group '{label}' to determine radius reliably.")
            return None, (None, None) # Return None radius and None center

        xg, yg = group[:, 0], group[:, 1]

        # --- Centering Correction: Calculate centroid ---
        center_x = np.mean(xg)
        center_y = np.mean(yg)
        # Calculate radial distance from the calculated centroid
        r = np.sqrt((xg - center_x)**2 + (yg - center_y)**2)
        # --- End Correction ---

        center_coords = (center_x, center_y)
        radius = None
        peak_indices = np.array([]) # Initialize

        try:
            # Compute radial histogram
            hist_r, bins_r = np.histogram(r, bins=nbins)
            centers_r = (bins_r[:-1] + bins_r[1:]) / 2

            # Peak detection in radial histogram
            max_hist_val = np.max(hist_r) if hist_r.size > 0 else 0
            if max_hist_val > 0:
                # Find peaks based on relative height
                peak_indices, properties = find_peaks(hist_r, height=max_hist_val * r_peak_height_fraction)

                if len(peak_indices) > 0:
                    # Select the *highest* peak as the most likely ring radius
                    highest_peak_index_in_list = np.argmax(properties["peak_heights"])
                    peak_index = peak_indices[highest_peak_index_in_list]
                    radius = centers_r[peak_index]
                # else: radius remains None
            else:
                 print(f"Warning: Radial histogram for '{label}' is empty or flat. Cannot find peaks.")

        except Exception as e:
            print(f"Error during radial analysis for '{label}': {e}")
            # Return None radius but keep calculated center if available
            return None, center_coords


        if plot:
            plt.figure(figsize=(8, 5))
            plt.plot(centers_r, hist_r, label='Ring')
            if radius is not None:
                plt.axvline(radius, color='r', linestyle='--', label=f'Highest Peak Radius: {radius:.2f} nm')
                # Optionally plot all found peaks for inspection
                if len(peak_indices) > 0:
                   plt.plot(centers_r[peak_indices], hist_r[peak_indices], "x", color='g', markersize=8, label='All Detected Peaks')

            plt.title(f'Radial Histogram \nCentroid (xy): ({center_x:.2f}, {center_y:.2f}) nm')
            plt.xlabel('Radius from Centroid (nm)')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            if save_to:
                if os.path.exists(save_to):
                    save_to = save_to.split('.')[0] + 'next' + '.png'
                plt.savefig(save_to, dpi=600)
                print(f"Plots saved to: {save_to}")
                plt.show()
            else:
                plt.show()

        return radius, center_coords

    # Calculate radius and center for lower and upper halves
    radius_lower, center_lower = get_radial_hist_and_peak(lower_half, 'Lower Half', nbins, r_peak_height_fraction, plot, save_to)
    radius_upper, center_upper = get_radial_hist_and_peak(upper_half, 'Upper Half', nbins, r_peak_height_fraction, plot, save_to)

    return radius_lower, radius_upper, center_lower, center_upper, z_split

def get_all_samples(folder):
    from helpers.readers import read_by_sample
    import glob
    import os

    files = glob.glob(os.path.join(folder,'*output*'))
    len_files = len(files)

    samples = [read_by_sample(folder, str(i)) for i in range(0, len_files)]

    return samples

def remove_outliers(points, factor=1.5):
    """Remove outliers using the IQR method on each dimension independently"""
    mask = np.ones(len(points), dtype=bool)  # Start with all points included

    for dim in range(3):  # For each dimension (x, y, z)
        q1 = np.percentile(points[:, dim], 25)
        q3 = np.percentile(points[:, dim], 75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        # Update the mask (AND operation with current dimension)
        dim_mask = (points[:, dim] >= lower_bound) & (points[:, dim] <= upper_bound)
        mask = mask & dim_mask

    # Apply the combined mask only once at the end
    return points[mask]

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


def plot_errors_by_corners(data_dict, save_to=None):
    """
    Plot L1 errors based on number of corners removed

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing 'l1_errors', 'corner_labels', and 'corner_names'
    """
    # Extract relevant data
    l1_errors = data_dict['l1_errors']
    corner_labels = data_dict['corner_labels']
    corner_names = data_dict['corner_names']

    # Create a DataFrame for easy plotting
    df = pd.DataFrame({
        'l1_error': l1_errors,
        'corner_count': corner_labels,
        'corner_description': corner_names
    })

    # Group by corner count and compute statistics
    stats_df = df.groupby('corner_count')['l1_error'].agg(['mean', 'std', 'count']).reset_index()

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Box plot to show distribution
    ax = sns.boxplot(x='corner_count', y='l1_error', data=df, palette='Set3')

    # Add individual points for better visualization
    sns.stripplot(x='corner_count', y='l1_error', data=df,
                  size=4, color=".3", linewidth=0, alpha=0.6)

    # Add error bars for mean values
    sns.pointplot(x='corner_count', y='mean', data=stats_df,
                  color='darkred', scale=0.7, markers='D')

    # Enhance the plot
    plt.title('CD by Number of Corners Removed', fontsize=15)
    plt.xlabel('Number of Corners Removed', fontsize=12)
    plt.ylabel('CD', fontsize=12)

    # Add count information
    for i, row in stats_df.iterrows():
        plt.text(row['corner_count'], row['mean'],
                 f"n={int(row['count'])}",
                 ha='center', va='bottom', color='darkred')

    plt.tight_layout()
    plt.show()

    # Alternative: Create a violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='corner_count', y='l1_error', data=df, palette='Set2', inner='quartile')
    plt.title('Distribution of CD by Number of Corners Removed', fontsize=15)
    plt.xlabel('Number of Corners Removed', fontsize=12)
    plt.ylabel('CD Error', fontsize=12)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=600)
    plt.show()



    # Create a relationship plot between corner count and error
    plt.figure(figsize=(10, 6))
    sns.regplot(x='corner_count', y='l1_error', data=df,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
    plt.title('Relationship Between Corner Count and CD Error', fontsize=15)
    plt.xlabel('Number of Corners Removed', fontsize=12)
    plt.ylabel('CD Error', fontsize=12)
    plt.tight_layout()
    plt.show()


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
#axes[0, 0].axvline(np.mean(num_points), color='red', linestyle='dashed', label='Mean')
axes[0, 0].set_title("Number of Points")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_ylim(bottom=0)

# Histogram with density plot for cluster_sizes
sns.histplot(cluster_sizes, kde=True, bins=50, ax=axes[0, 1], color="teal")
#axes[0, 1].axvline(np.mean(cluster_sizes), color='red', linestyle='dashed', label='Mean')
axes[0, 1].set_title("Cluster Sizes")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_ylim(bottom=0)

# Boxplot for num_points
sns.boxplot(y=num_points, ax=axes[1, 0], color="teal")
axes[1, 0].set_title("Number of Points")

# Boxplot for cluster_sizes
sns.boxplot(y=cluster_sizes, ax=axes[1, 1], color="teal")
axes[1, 1].set_title("Cluster Sizes")

# Adjust layout
plt.tight_layout()
plt.savefig('/home/dim26fa/figures/sim_stats_num_points1804.png', dpi=600)
plt.show()