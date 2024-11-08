import numpy as np
import open3d.cuda.pybind.io
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import hdbscan
import scipy


# Functions to get statistics about datasets - e.g. about experimental datasets
def analyze_point_cloud(point_cloud, min_cluster_size):
    """
    Method to analyze point cloud statistics.
    :param point_cloud: point cloud of shape (N, 3)
    :param min_cluster_size: minimum number of points in a cluster, parameter for HDBSCAN
    :return: dict containing statistics of point cloud
    """
    # Calculate the total number of points
    num_points = len(point_cloud)

    # Calculate the centroid
    centroid = point_cloud.mean()

    # Calculate the standard deviation
    std_dev = point_cloud.std()

    var = point_cloud.var()
    # Calculate the bounding box size (max - min for each dimension)
    min_vals = point_cloud.min()
    max_vals = point_cloud.max()
    bounding_box_size = max_vals - min_vals

    # Calculate eigenvalues and eigenvectors for shape analysis
    covariance_matrix = point_cloud.cov()
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Cluster the points using HDBSCAN
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size)
    cluster_labels = hdbscan_clusterer.fit_predict(point_cloud)

    # Identify unique clusters (excluding noise points labeled as -1)
    unique_clusters = set(cluster_labels) - {-1}

    # Calculate cluster centroids and distances between them
    centroids = []
    cluster_sizes = []
    for cluster in unique_clusters:
        cluster_points = point_cloud[cluster_labels == cluster]
        cluster_sizes.append(len(cluster_points))
        centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    cluster_distances = calculate_centroid_distances(centroids)

    return {
        'num_points': num_points,
        'centroid': centroid,
        'std_dev': std_dev,
        'var': var,
        'bounding_box_size': bounding_box_size,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'cluster_distances': cluster_distances,
        'cluster_sizes': cluster_sizes
    }


def analyze_point_cloud_clusters(point_cloud, min_cluster_size):
    """
    Method to analyze clusters in a point cloud.
    :param point_cloud: point cloud of shape (N, 3)
    :param min_cluster_size: minimum number of points in a cluster, parameter for HDBSCAN
    :return: dict containing statistics for each cluster in the point cloud
    """
    # Cluster the points using HDBSCAN
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = hdbscan_clusterer.fit_predict(point_cloud)

    # Initialize dictionary to hold statistics for each cluster
    cluster_stats = {}

    # Identify unique clusters (excluding noise points labeled as -1)
    unique_clusters = set(cluster_labels) - {-1}

    for cluster in unique_clusters:
        # Extract points for this cluster
        cluster_points = point_cloud[cluster_labels == cluster]

        # Calculate statistics for this cluster
        num_points = len(cluster_points)
        centroid = cluster_points.mean(axis=0)
        std_dev = cluster_points.std(axis=0)
        var = cluster_points.var(axis=0)
        min_vals = cluster_points.min(axis=0)
        max_vals = cluster_points.max(axis=0)
        bounding_box_size = max_vals - min_vals

        # Eigenvalues and eigenvectors
        covariance_matrix = np.cov(cluster_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Add statistics to the cluster stats dictionary
        cluster_stats[cluster] = {
            'num_points': num_points,
            'centroid': centroid,
            'std_dev': std_dev,
            'var': var,
            'bounding_box_size': bounding_box_size,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors
        }

    return cluster_stats


def calculate_centroid_distances(centroids):
    """
    Calculate distances between cluster centroids.
    :param centroids: array of centroid coordinates of shape (k, 3)
    :return: list of distances between each pair of centroids
    """
    num_centroids = len(centroids)
    distances = []
    for i in range(num_centroids):
        for j in range(i + 1, num_centroids):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            distances.append(distance)
    return distances


def load_and_analyze_point_clouds(directory, suffix='.csv', min_cluster_size=25, analyze='cluster'):
    """
    Load point clouds from a directory and analyze them.
    :param directory: path to directory containing point clouds
    :param suffix: suffix that the point clouds should have to be read
    :return: dict of statistics
    """
    stats_list = []
    if suffix == '.csv':
        file_names = [f for f in os.listdir(directory) if f.endswith('.csv')]
    elif suffix == '.ply':
        file_names = [f for f in os.listdir(directory) if f.endswith('.csv')]
    else:
        raise ValueError('Suffix must be .csv or .ply files')

    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        if suffix == '.csv':
            point_cloud = pd.read_csv(file_path)
            point_cloud.rename(columns=lambda col: col.replace("nm", "") if "nm" in col else col, inplace=True)
        elif suffix == '.ply':
            import open3d
            pc = open3d.io.read_point_cloud(file_path)
            point_cloud = pd.DataFrame(pc.points, columns=['x', 'y', 'z'])
        else:
            raise ValueError('Suffix must be .csv or .ply files')
        if analyze=='cluster':
            stats = analyze_point_cloud_clusters(point_cloud, min_cluster_size)
        else:
            stats = analyze_point_cloud(point_cloud, min_cluster_size)
        stats_list.append(stats)

    return stats_list


def plot_number_points(stats, style='dark', color='steelblue', save=False):
    """
    Plot number of points in the point clouds in a specific folder as a distribution.
    :param stats: the dict file containing statistics
    :param style: choose between 'white' and 'dark'
    :param color: choose the color of the chart
    :param save: bool save the figure
    :return:
    """

    if style == 'dark':
        sns.set_style("dark", {'axes.facecolor': 'black', 'figure.facecolor': 'black'})
        clr = 'white'
    else:
        sns.set_style("white")
        clr = 'black'
    num_points = [stats['num_points'] for stats in stats]
    sns.histplot(num_points, bins=20, kde=True, color=color)  # KDE (Kernel Density Estimate) adds a smoothed line to help understand the distribution shape
    plt.title('Distribution of Number of Points', color=clr)
    plt.xlabel('Number of Points', color=clr)
    plt.ylabel('Frequency', color=clr)
    plt.tick_params(colors=clr)
    if save:
        plt.savefig('/home/dim26fa/graphics/smlms/points.png', dpi=300)
    plt.show()


def plot_bounding_box_size(stats, style='dark', color='steelblue'):
    if style == 'dark':
        sns.set_style("dark", {'axes.facecolor': 'black', 'figure.facecolor': 'black'})
        clr = 'white'
    else:
        sns.set_style("white")
        clr = 'black'
    bounding_box_sizes_df = pd.DataFrame([stats['bounding_box_size'] for stats in stats], columns=['x', 'y', 'z'])
    # Melt the DataFrame to long-format for seaborn's boxplot
    bounding_box_sizes_melted = bounding_box_sizes_df.melt(var_name='Dimension', value_name='Size')

    sns.boxplot(x='Dimension', y='Size', data=bounding_box_sizes_melted,
                color=color,  # Color of the box fill
                linewidth=1,    # Thickness of the lines
                fliersize=5,     # Size of the outlier markers
                flierprops={'markerfacecolor':clr, 'markeredgecolor':clr},  # Outlier styles
                medianprops={'color':clr},  # Median line style
                whiskerprops={'color':clr},  # Whisker line style
                capprops={'color':clr},      # Cap line style
                boxprops={'edgecolor':clr})  # Box edge line style)
    plt.title('Bounding Box Size Distribution', color=clr)
    if style=='dark':
        plt.tick_params(colors=clr)
    plt.xlabel('Dimension', color=clr)
    plt.ylabel('Size', color=clr)
    plt.show()


def plot_standard_dev(stats, style='dark', color='steelblue', save=False):
    if style=='dark':
        sns.set_style("dark", {'axes.facecolor': 'black', 'figure.facecolor': 'black'})
        clr = 'white'
    else:
        sns.set_style("white")
        clr='black'
    std_dev_df = pd.DataFrame([stats['std_dev'] for stats in stats], columns=['x', 'y', 'z'])

    std_dev_melted = std_dev_df.melt(var_name='Dimension', value_name='Standard Deviation')

    sns.boxplot(x='Dimension', y='Standard Deviation', data=std_dev_melted,
                color=color,
                linewidth=1, flierprops={'markerfacecolor':clr, 'markeredgecolor':clr},
                medianprops={'color':clr}, whiskerprops={'color':clr},
                capprops={'color':clr}, boxprops={'edgecolor':clr})

    plt.title('Standard Deviation Distribution by Dimension', color=clr)
    plt.xlabel('Dimension', color=clr)
    plt.ylabel('Standard Deviation', color=clr)
    plt.tick_params(colors=clr)
    if save:
        plt.savefig('/home/dim26fa/graphics/smlms/std_dev.png', dpi=300)
    plt.show()


def plot_eigenvalues(stats, style='dark', color='steelblue'):
    if style=='dark':
        sns.set_style("dark", {'axes.facecolor': 'black', 'figure.facecolor': 'black'})
        clr = 'white'
    else:
        sns.set_style("white")
        clr='black'
    eigenvalues_df = pd.DataFrame([stats['eigenvalues'] for stats in stats], columns=['1st Principal', '2nd Principal', '3rd Principal'])
    eigenvalues_melted = eigenvalues_df.melt(var_name='Principal Component', value_name='Eigenvalue')

    sns.boxplot(x='Principal Component', y='Eigenvalue', data=eigenvalues_melted,
                color=color,
                linewidth=1, flierprops={'markerfacecolor':clr, 'markeredgecolor':clr},
                medianprops={'color':clr}, whiskerprops={'color':clr},
                capprops={'color':clr}, boxprops={'edgecolor':clr})

    plt.title('Eigenvalue Distribution by Principal Component', color=clr)
    plt.xlabel('Principal Component', color=clr)
    plt.ylabel('Eigenvalue', color=clr)
    plt.tick_params(colors=clr)
    plt.show()


def plot_xyz_distribution(df, style='dark', palette=None, save=False):
    if palette is not None:
        sns.set_palette(palette)
        current_palette = sns.color_palette()
    else:
        sns.set_palette(sns.color_palette())
        current_palette = sns.color_palette()
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
    sns.histplot(df['x'], kde=True, color = current_palette[0])
    plt.title('Distribution of Points in X Direction', color=clr)
    plt.xlabel('x', color=clr)
    plt.ylabel('Count', color=clr)
    plt.tick_params(colors=clr)

    plt.subplot(1, 3, 2)
    sns.histplot(df['y'], kde=True, color = current_palette[1])
    plt.title('Distribution of Points in Y Direction', color=clr)
    plt.xlabel('y', color=clr)
    plt.ylabel('Count', color=clr)
    plt.tick_params(colors=clr)

    plt.subplot(1, 3, 3)
    sns.histplot(df['z'], kde=True, color = current_palette[2])
    plt.title('Distribution of Points in Z Direction', color=clr)
    plt.xlabel('z', color=clr)
    plt.ylabel('Count', color=clr)
    plt.tick_params(colors=clr)
    plt.tight_layout()
    if save:
        plt.savefig('/home/dim26fa/graphics/smlms/xyz_distrib_sample.png', dpi=300)
    plt.show()


def plot_xyz_distribution_boxplot(df, style='dark', color='steelblue'):
    # Setting the background style for better visibility
    if style=='dark':
        sns.set_style("dark", {'axes.facecolor': 'black', 'figure.facecolor': 'black'})
        clr = 'white'
    else:
        sns.set_style("white")
        clr = 'black'
    # Plotting box plots
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df,
                color=color,
                linewidth=1, flierprops={'markerfacecolor':clr, 'markeredgecolor':clr},
                medianprops={'color':clr}, whiskerprops={'color':clr},
                capprops={'color':clr}, boxprops={'edgecolor':clr})
    plt.title('Box Plot of Point Distribution in X, Y, Z Directions', color=clr)
    plt.ylabel('Value', color=clr)
    plt.show()


def plot_variance(stats, style='dark', color='steelblue', save=False):
    """
    Plot variance distribution by dimension.

    :param stats: List of dictionaries with 'var' key containing variance data for each point cloud.
    :param style: Seaborn style for the plot.
    :param color: Color of the boxplot.
    :param save: Whether to save the plot or not.
    """
    if style == 'dark':
        sns.set_style("dark", {'axes.facecolor': 'black', 'figure.facecolor': 'black'})
        clr = 'white'
    else:
        sns.set_style("white")
        clr = 'black'

    # Extract variance data from the stats
    var_df = pd.DataFrame([s['var'] for s in stats], columns=['x', 'y', 'z'])

    # Melt the DataFrame for seaborn boxplot
    var_melted = var_df.melt(var_name='Dimension', value_name='Variance')

    # Create the boxplot
    sns.boxplot(x='Dimension', y='Variance', data=var_melted,
                color=color,
                linewidth=1, flierprops={'markerfacecolor': clr, 'markeredgecolor': clr},
                medianprops={'color': clr}, whiskerprops={'color': clr},
                capprops={'color': clr}, boxprops={'edgecolor': clr})

    # Add titles and labels
    plt.title('Variance Distribution by Dimension', color=clr)
    plt.xlabel('Dimension', color=clr)
    plt.ylabel('Variance', color=clr)
    plt.tick_params(colors=clr)

    # Save the plot if requested
    if save:
        directory = '/home/dim26fa/graphics/smlms/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, 'variance.png'), dpi=300)

    # Show the plot
    plt.show()


def plot_xyz_distribution_multiple_samples(data_list, sample_names=None, style='dark', palette=None):
    if palette is not None:
        sns.set_palette(palette)
        current_palette = sns.color_palette()
    else:
        sns.set_palette(sns.color_palette())
        current_palette = sns.color_palette()

    # Setting the background style for better visibility
    if style == 'dark':
        sns.set_style("dark", {'axes.facecolor': 'black', 'figure.facecolor': 'black'})
        clr = 'white'
    else:
        sns.set_style("white")
        clr = 'black'

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # If sample names are not provided, create default names
    if sample_names is None:
        sample_names = [f"Sample {i + 1}" for i in range(len(data_list))]

    # Plot histograms for each sample
    for i, df in enumerate(data_list):
        color = current_palette[i % len(current_palette)]

        sns.histplot(df['x'], kde=True, color=color, alpha=0.5, ax=axes[0], label=sample_names[i])
        sns.histplot(df['y'], kde=True, color=color, alpha=0.5, ax=axes[1], label=sample_names[i])
        sns.histplot(df['z'], kde=True, color=color, alpha=0.5, ax=axes[2], label=sample_names[i])

    # Customize each subplot
    for ax, direction in zip(axes, ['X', 'Y', 'Z']):
        ax.set_title(f'Distribution of Points in {direction} Direction', color=clr)
        ax.set_xlabel(direction.lower(), color=clr)
        ax.set_ylabel('Count', color=clr)
        ax.tick_params(colors=clr)
        ax.legend(title='Samples', title_fontsize='10', fontsize='8')

    plt.tight_layout()
    plt.show()


def calculate_distribution_from_multiple_csvs(csv_file_paths, x_col='x', y_col='y', z_col='z'):
    """
    Read multiple CSV files, extract x, y, z coordinates, and calculate the overall distribution.

    :param csv_file_paths: List of paths to the CSV files
    :param x_col: Name of the column containing x coordinates (default 'x')
    :param y_col: Name of the column containing y coordinates (default 'y')
    :param z_col: Name of the column containing z coordinates (default 'z')
    :return: Dictionary containing distribution parameters for x, y, and z
    """
    all_coords = []

    for file_path in csv_file_paths:
        # Read each CSV file
        df = pd.read_csv(file_path)

        # Extract x, y, z coordinates and add to the list
        coords = df[[x_col, y_col, z_col]].values
        all_coords.append(coords)

    # Combine all coordinates into a single array
    all_coords = np.vstack(all_coords)

    # Separate x, y, and z coordinates
    x_coords = all_coords[:, 0]
    y_coords = all_coords[:, 1]
    z_coords = all_coords[:, 2]

    # Calculate mean and standard deviation for each dimension
    x_mean, x_std = np.mean(x_coords), np.std(x_coords)
    y_mean, y_std = np.mean(y_coords), np.std(y_coords)
    z_mean, z_std = np.mean(z_coords), np.std(z_coords)

    # Perform Shapiro-Wilk test to check for normality
    _, x_p_value = scipy.stats.shapiro(x_coords)
    _, y_p_value = scipy.stats.shapiro(y_coords)
    _, z_p_value = scipy.stats.shapiro(z_coords)

    # Determine if distributions are normal (you can adjust the threshold)
    normal_threshold = 0.05
    x_is_normal = x_p_value > normal_threshold
    y_is_normal = y_p_value > normal_threshold
    z_is_normal = z_p_value > normal_threshold

    return {
        'x': {'mean': x_mean, 'std': x_std, 'is_normal': x_is_normal},
        'y': {'mean': y_mean, 'std': y_std, 'is_normal': y_is_normal},
        'z': {'mean': z_mean, 'std': z_std, 'is_normal': z_is_normal}
    }

