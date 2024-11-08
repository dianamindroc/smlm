import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from scipy.spatial import procrustes
from lets_plot import *
import os

def read_and_display_2d_data(file_path):
    # Read only the first two columns (x and y) from the .txt file with whitespace as a separator, and skip the header row
    data = pd.read_csv(file_path, delim_whitespace=True, usecols=[0, 1], skiprows=1, header=None, names=['x', 'y'])
    x, y = data['x'], data['y']

    # Display the data as a 2D scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=1, c='black', alpha=0.5)
    plt.title('2D SMLM Data')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

    return data

def read_and_display_3d_data(file_path):
    # Read the first three columns (x, y, z) from the .txt file with whitespace as a separator, and skip the header row
    data = pd.read_csv(file_path, delim_whitespace=True, usecols=[0, 1, 2], skiprows=1, header=None, names=['x', 'y', 'z'])
    x, y, z = data['x'], data['y'], data['z']

    # Display the data as a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1, c='black', alpha=0.5)
    ax.set_title('3D SMLM Data')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    plt.show()

    return data

def hdbscan_clustering_and_display(data, min_cluster_size=10, min_samples=20, cmap='tab20'):
    # Apply HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(data[['x', 'y']])

    # Filter out noise (label == -1 is considered noise in HDBSCAN) and create a copy
    clustered_data = data[labels != -1].copy()
    clustered_data['cluster'] = labels[labels != -1]

    # Plot clusters with a non-gradient colormap
    plt.figure(figsize=(8, 6))
    plt.scatter(clustered_data['x'], clustered_data['y'], c=clustered_data['cluster'], cmap=cmap, s=1)
    plt.title('HDBSCAN Clustering (Excluding Noise)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.colorbar(label='Cluster Label')
    plt.show()

    return clustered_data


def plot_selected_clusters(data, cluster_ids):
    """
    Plots specified clusters from the data.

    Parameters:
    - data: DataFrame containing 'x', 'y', and 'cluster' columns.
    - cluster_ids: List of cluster IDs to visualize.
    """
    plt.figure(figsize=(10, 8))

    for cluster_id in cluster_ids:
        # Extract points for this cluster
        cluster_points = data[data['cluster'] == cluster_id]

        # Plot the cluster if it exists in the data
        if not cluster_points.empty:
            plt.scatter(cluster_points['x'], cluster_points['y'], label=f'Cluster {cluster_id}', s=10)
        else:
            print(f"Warning: Cluster {cluster_id} has no points.")

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Selected Clusters in SMLM Data')
    plt.legend()
    plt.xlim(data['x'].min(), data['x'].max())
    plt.ylim(data['y'].min(), data['y'].max())
    plt.grid()
    plt.show()

def plot_selected_clusters_3d(data, cluster_ids):
    """
    Plots specified clusters from the data in 3D.

    Parameters:
    - data: DataFrame containing 'x', 'y', 'z', and 'cluster' columns.
    - cluster_ids: List of cluster IDs to visualize.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster_id in cluster_ids:
        # Extract points for this cluster
        cluster_points = data[data['cluster'] == cluster_id]

        # Plot the cluster if it exists in the data
        if not cluster_points.empty:
            ax.scatter(cluster_points['xnm'], cluster_points['ynm'], cluster_points['znm'],
                       label=f'Cluster {cluster_id}', s=10)
        else:
            print(f"Warning: Cluster {cluster_id} has no points.")

    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('Z (nm)')
    ax.set_title('Selected Clusters in SMLM Data - 3D')
    ax.legend()
    ax.grid()
    plt.show()


def save_and_plot_clusters_3d(data, output_dir='clusters'):
    """
    Saves each specified cluster to a separate file and optionally plots it.

    Parameters:
    - data: DataFrame containing 'xnm', 'ynm', 'znm', and 'cluster' columns.
    - cluster_ids: List of cluster IDs to save and plot.
    - output_dir: Directory to save each cluster file.
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    cluster_ids = data['cluster'].unique()

    for cluster_id in cluster_ids:
        # Extract points for this cluster
        cluster_points = data[data['cluster'] == cluster_id]

        if not cluster_points.empty:
            # Save the cluster data as a CSV file
            filename = os.path.join(output_dir, f'cluster_{cluster_id}.csv')
            cluster_points.to_csv(filename, index=False)
            print(f"Cluster {cluster_id} saved to {filename}.")

            # Plotting the cluster
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(cluster_points['xnm'], cluster_points['ynm'], cluster_points['znm'], s=10)
            ax.set_xlabel('X (nm)')
            ax.set_ylabel('Y (nm)')
            ax.set_zlabel('Z (nm)')
            ax.set_title(f'Cluster {cluster_id} in 3D')
            plt.show()
        else:
            print(f"Warning: Cluster {cluster_id} has no points and was not saved.")

def save_and_plot_clusters_2d(data, cluster_ids, output_dir='clusters', plot=False):
    """
    Saves each specified 2D cluster to a separate file and optionally plots it.

    Parameters:
    - data: DataFrame containing 'x', 'y', and 'cluster' columns.
    - cluster_ids: List of cluster IDs to save and plot.
    - output_dir: Directory to save each cluster file.
    - plot: Boolean indicating whether to display a plot for each cluster.
    """
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    for cluster_id in cluster_ids:
        # Extract points for this cluster
        cluster_points = data[data['cluster'] == cluster_id]

        if not cluster_points.empty:
            # Prepare the base filename
            base_filename = os.path.join(output_dir, f'cluster_{cluster_id}.csv')
            filename = base_filename

            # Check if a file with this name already exists, and add underscores if needed
            counter = 0
            while os.path.exists(filename):
                counter += 1
                filename = os.path.join(output_dir, f'cluster_{cluster_id}{"_" * counter}.csv')

            # Save the cluster data as a CSV file
            cluster_points.to_csv(filename, index=False)
            print(f"Cluster {cluster_id} saved to {filename}.")

            # Plotting the cluster if plot is True
            if plot:
                fig, ax = plt.subplots()
                ax.scatter(cluster_points['x'], cluster_points['y'], s=10)
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                ax.set_title(f'Cluster {cluster_id} in 2D')
                plt.show()
        else:
            print(f"Warning: Cluster {cluster_id} has no points and was not saved.")

def procrustes_disparity(ref_points, candidate_points):
    # Run Procrustes analysis to get disparity (similarity score)
    mtx1, mtx2, disparity = procrustes(ref_points, candidate_points)
    return disparity

def detect_npcs_with_clustering_and_matching(data_coords, reference_templates, hdbscan_params, match_threshold):
    # Step 1: Cluster the aggregated points
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    labels = clusterer.fit_predict(data_coords)

    matched_npcs = []

    # Step 2: Iterate over each cluster for template matching
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip noise points

        # Extract candidate cluster points
        candidate_points = data_coords[labels == cluster_id]

        # Step 3: Match each candidate with the reference templates
        for ref_template in reference_templates:
            disparity = procrustes_disparity(ref_template, candidate_points)
            if disparity < match_threshold:
                matched_npcs.append(candidate_points)
                break  # Stop if a match is found for this cluster

    return matched_npcs


def plot_point_clouds(folder_path, max_subplots=28, save=True, plot=False):
    # List of all CSV files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    num_files = len(files)

    # Define maximum subplots per figure and subplot grid dimensions
    max_subplots = max_subplots
    n_rows = 4
    n_cols = 7
    figg=0
    # Loop over files in batches of 28
    for i in range(0, num_files, max_subplots):
        # Set up a new figure for each batch of 28 point clouds
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 8))
        fig.suptitle(f"Point Clouds {i + 1} to {min(i + max_subplots, num_files)}", fontsize=18)

        # Flatten the axes array for easier indexing
        axs = axs.flatten()

        # Process each file in the current batch
        for j, file_name in enumerate(files[i:i + max_subplots]):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Ensure the file has at least two columns (X, Y coordinates)
            if df.shape[1] < 2:
                print(f"File {file_name} has insufficient data for plotting.")
                continue

            # Extract the X and Y coordinates from the first two columns
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]

            # Plot in the corresponding subplot
            axs[j].scatter(x, y, s=1)  # Adjust point size as needed
            axs[j].set_title(file_name, fontsize=8)

            #axs[j].axis('off')  # Turn off axis for cleaner visualization
            # Manually remove ticks and spines
            axs[j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in axs[j].spines.values():
                spine.set_visible(False)

        # Turn off any unused subplots in the current figure
        for k in range(j + 1, max_subplots):
            axs[k].axis('off')

        # Display the current figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)# Adjust layout to fit title
        if save:
            plt.savefig(folder_path + '/plot_' + str(figg) + '.jpg')
        if plot:
            plt.show()
        figg+=1


def plot_point_clouds_hist(folder_path, max_subplots=28, grid_size=0.01, cmap='hot', save=True, plot=False):
    # List of all CSV files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    num_files = len(files)

    # Define maximum subplots per figure and subplot grid dimensions
    n_rows = 4
    n_cols = 7
    fig_counter = 0

    # Loop over files in batches of 28
    for i in range(0, num_files, max_subplots):
        # Set up a new figure for each batch of 28 point clouds
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 8))
        fig.suptitle(f"Point Clouds {i + 1} to {min(i + max_subplots, num_files)}", fontsize=18)

        # Flatten the axes array for easier indexing
        axs = axs.flatten()

        # Store the maximum density value for consistent scaling
        vmax = 0
        histograms = []  # Store histograms and extents for each plot

        # Process each file in the current batch
        for j, file_name in enumerate(files[i:i + max_subplots]):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)

            # Ensure the file has at least two columns (X, Y coordinates)
            if df.shape[1] < 2:
                print(f"File {file_name} has insufficient data for plotting.")
                continue

            # Extract the X and Y coordinates from the first two columns
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values

            # Compute 2D histogram for intensity mapping
            hist, xedges, yedges = np.histogram2d(x, y, bins=int(1 / grid_size))
            histograms.append((hist, xedges, yedges))  # Store for consistent scaling

            # Update vmax for consistent scaling across subplots
            vmax = max(vmax, hist.max())

        # Plot each histogram with consistent vmax scaling
        for j, (hist, xedges, yedges) in enumerate(histograms):
            im = axs[j].imshow(hist.T, origin='lower', cmap=cmap, interpolation='nearest',
                               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=0, vmax=vmax)
            axs[j].set_title(files[i + j], fontsize=8)
            axs[j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in axs[j].spines.values():
                spine.set_visible(False)

        # Turn off any unused subplots in the current figure
        for k in range(j + 1, max_subplots):
            axs[k].axis('off')

        # Adjust layout and save or display figure
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        if save:
            plt.savefig(f'/home/dim26fa/data/NPC_Sauer_Danush/3D_Biplane/clustering/plot_{fig_counter}.jpg', dpi=300)
        if plot:
            plt.show()

        fig_counter += 1

def remove_column_from_csvs(folder_path, column_name='clusters'):
    # Loop through all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):  # Process only CSV files
            file_path = os.path.join(folder_path, filename)

            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Check if the specified column exists and remove it
            if column_name in df.columns:
                df.drop(columns=[column_name], inplace=True)

                # Save the updated DataFrame back to the same file
                df.to_csv(file_path, index=False)
                print(f"Updated and saved: {filename}")
            else:
                print(f"No '{column_name}' column found in: {filename}")

def plot_single_2d_histogram(data, grid_size=0.01, cmap='hot', title=None, save=False):
    """
    Generates a 2D histogram projection of a single point cloud.

    :param data: Either a file path to the CSV file containing the 2D point cloud data
                 (expects at least two columns: X, Y) or a NumPy array/Pandas DataFrame
                 with X and Y coordinates in the first two columns.
    :param grid_size: The size of each grid cell in the histogram, default is 0.01.
    :param cmap: Colormap for the intensity image, default is 'hot'.
    :param title: Title for the plot. Optional.
    :param save: Boolean indicating whether to save the plot as an image file. Default is False.
    """
    # Load the point cloud data from file if a path is given
    if isinstance(data, str):
        df = pd.read_csv(data)
        if df.shape[1] < 2:
            print("The file does not contain sufficient data for plotting.")
            return
        x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
    elif isinstance(data, (np.ndarray, pd.DataFrame)):
        if data.shape[1] < 2:
            print("The array does not contain sufficient data for plotting.")
            return
        x, y = data[:, 0], data[:, 1] if isinstance(data, np.ndarray) else data.iloc[:, 0], data.iloc[:, 1]
    else:
        print("Input data must be a file path, NumPy array, or Pandas DataFrame.")
        return

    # Define the range for histogram to be centered and consistent
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    max_range = max(x_max - x_min, y_max - y_min)

    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_min, x_max = x_center - max_range / 2, x_center + max_range / 2
    y_min, y_max = y_center - max_range / 2, y_center + max_range / 2

    # Compute the number of bins
    num_bins = int(max_range / grid_size) + 1

    # Compute 2D histogram for intensity mapping
    hist, xedges, yedges = np.histogram2d(x, y, bins=num_bins, range=[[x_min, x_max], [y_min, y_max]])

    hist_normalized = hist.T / np.max(hist)

    # Plot the histogram as an image
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(hist_normalized, origin='lower', cmap=cmap, interpolation='nearest',
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    # Customize plot appearance
    if title is not None:
        ax.set_title(title, fontsize=16)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    fig.colorbar(im, ax=ax, label="Point Density")

    # Optional: save the plot
    if save:
        save_dir = os.path.join(os.path.dirname(data) if isinstance(data, str) else '.', 'plots')
        os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.basename(data) if isinstance(data, str) else 'array_data'
        save_path = os.path.join(save_dir, file_name.replace('.csv', '_histogram.png'))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def center_clusters_in_folder(folder_path, output_folder=None):
    """
    Reads all CSV files from the specified folder, centers each cluster around (0, 0),
    and saves the results.

    Parameters:
        folder_path (str): Path to the folder containing CSV files.
        output_folder (str, optional): Path to save centered CSV files. If None, files
                                       will not be saved.

    Returns:
        dict: Dictionary containing centered data for each file.
    """
    centered_clusters = {}

    # Create output folder if saving and it doesn't exist
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)

            # Read the CSV file
            cluster_data = pd.read_csv(file_path)

            # Assume columns 'x' and 'y' represent coordinates
            if 'x' in cluster_data.columns and 'y' in cluster_data.columns:
                # Center the cluster around (0, 0)
                cluster_data['x'] = cluster_data['x'] - cluster_data['x'].mean()
                cluster_data['y'] = cluster_data['y'] - cluster_data['y'].mean()

                # Save the centered data in the dictionary
                centered_clusters[filename] = cluster_data

                # If an output folder is specified, save the centered cluster to a new CSV
                if output_folder:
                    output_path = os.path.join(output_folder, f"centered_{filename}")
                    cluster_data.to_csv(output_path, index=False)

            else:
                print(f"Skipping {filename}: does not contain 'x' and 'y' columns.")

    return centered_clusters


def analyze_samples_diameters(folder_path, tolerance_percentage=10):
    from pathlib import Path
    """
    Analyze all CSV files in a folder and group them by similar diameters.

    Parameters:
    folder_path: str or Path - path to folder containing CSV files
    tolerance_percentage: float - percentage difference allowed to consider diameters similar

    Returns:
    dict - groups of files with similar diameters
    """
    # Get all CSV files
    folder_path = Path(folder_path)
    csv_files = list(folder_path.glob('*.csv'))

    # Store results
    diameter_results = {}

    # Analyze each file
    for file_path in csv_files:
        try:
            # Read data
            df = pd.read_csv(file_path)
            x, y = df.iloc[:, 0].values, df.iloc[:, 1].values

            # Find centroid
            center_x = np.mean(x)
            center_y = np.mean(y)

            # Calculate distances from center
            distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Calculate diameter using 95th percentile
            radius_95 = np.percentile(distances, 95)
            diameter = 2 * radius_95

            diameter_results[file_path.name] = diameter

        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

    # Group similar diameters
    tolerance = tolerance_percentage / 100
    diameter_groups = {}
    processed_files = set()

    # Sort diameters for consistent grouping
    sorted_diameters = sorted(diameter_results.items(), key=lambda x: x[1])

    for i, (file_name, diameter) in enumerate(sorted_diameters):
        if file_name in processed_files:
            continue

        group = []
        base_diameter = diameter

        # Check all remaining files for similar diameters
        for other_file, other_diameter in sorted_diameters[i:]:
            if other_file not in processed_files:
                difference = abs(other_diameter - base_diameter) / base_diameter
                if difference <= tolerance:
                    group.append((other_file, other_diameter))
                    processed_files.add(other_file)

        if group:
            group_key = f"Group {len(diameter_groups) + 1} (Diameter ? {base_diameter:.2f})"
            diameter_groups[group_key] = group

    return diameter_groups