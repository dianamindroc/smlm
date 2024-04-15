import numpy as np
import open3d.cuda.pybind.io
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_point_cloud(point_cloud):
    """
    Method to analyze point cloud statistics.
    :param point_cloud: point cloud of shape (N, 3)
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

    return {
        'num_points': num_points,
        'centroid': centroid,
        'std_dev': std_dev,
        'var': var,
        'bounding_box_size': bounding_box_size,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }


def load_and_analyze_point_clouds(directory, suffix='.csv'):
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
        elif suffix == '.ply':
            import open3d
            pc = open3d.io.read_point_cloud(file_path)
            point_cloud = pd.DataFrame(pc.points, columns=['x', 'y', 'z'])
        else:
            raise ValueError('Suffix must be .csv or .ply files')
        stats = analyze_point_cloud(point_cloud)
        stats_list.append(stats)

    return stats_list


def plot_number_points(stats, style='dark', color='steelblue'):
    """
    Plot number of points in the point clouds in a specific folder as a distribution.
    :param stats: the dict file containing statistics
    :param style: choose between 'white' and 'dark'
    :param color: choose the color of the chart
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

def plot_standard_dev(stats, style='dark', color='steelblue'):
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

def plot_xyz_distribution(df, style='dark', palette=None):
    if palette is not None:
        sns.set_palette(palette)
        current_palette = palette
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
    plt.show()

def plot_xyz_distribution_boxplot(df, style, color):
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