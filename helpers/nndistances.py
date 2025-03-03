import numpy as np
import scipy.stats as stats
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

# Load all SMLM data files from a directory
def load_all_smlm_data(directory, file_extension="csv"):
    all_data = []
    file_paths = glob.glob(os.path.join(directory, f"*.{file_extension}"))  # Get all matching files

    for file in file_paths:
        data = pd.read_csv(file)
        data = np.array(data)  # Adjust delimiter if needed
        all_data.append(data[:, :3])  # Extract x, y, z columns

    return np.vstack(all_data)  # Combine all files into a single array


# Compute nearest neighbor distances
def compute_nnd(data):
    tree = cKDTree(data)
    distances, _ = tree.query(data, k=2)  # k=2 because the first neighbor is itself
    return distances[:, 1]  # Ignore the self-distance (first column)


# Fit a probability distribution
def fit_distribution(nnd, dist_name='gamma'):
    dist = getattr(stats, dist_name)  # Get the chosen distribution
    params = dist.fit(nnd)  # Fit distribution to data
    return dist, params


# Generate new samples from the fitted distribution
def sample_new_nnd(dist, params, size=500):
    return dist.rvs(*params, size=size)


# Plot histogram and fitted distribution
def plot_distribution(nnd, dist, params):
    plt.hist(nnd, bins=30, density=True, alpha=0.6, color='g', label="Empirical NND")

    x = np.linspace(min(nnd), max(nnd), 1000)
    pdf = dist.pdf(x, *params)
    plt.plot(x, pdf, 'r-', label="Fitted Distribution")

    plt.xlabel("Nearest Neighbor Distance")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.show()