def is_nested_dict(d):
    """
    Check if a variable is a nested dictionary.

    :param d: The variable to check.
    :return: True if it's a nested dictionary, False otherwise.
    """
    # First check if the variable itself is a dictionary
    if not isinstance(d, dict):
        return False

    # Check if any value in the dictionary is also a dictionary
    for value in d.values():
        if isinstance(value, dict):
            return True

    return False


def set_seed(seed=42):
    import random
    import numpy as np
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)  # For NumPy operations
    try:
        import torch
        torch.manual_seed(seed)  # If you're using PyTorch
    except ImportError:
        pass  # Skip if PyTorch is not installed

def extract_stats(stats):
    import pandas as pd
    """
    Extracts num_points, cluster_sizes, and standard deviations (cluster & sample)
    from a list of nested dictionaries.

    Parameters:
        stats (list): A list of dictionaries containing statistical data.

    Returns:
        dict: A dictionary containing extracted lists of values.
    """
    # Extract basic statistics
    num_points = [entry['num_points'] for item in stats for key, entry in item.items() if 'num_points' in entry]
    cluster_sizes = [entry['cluster_size'] for item in stats for key, entry in item.items() if 'cluster_size' in entry]
    cluster_stds = [entry['cluster_std'] for item in stats for key, entry in item.items() if 'cluster_std' in entry]
    sample_stds = [entry['sample_std'] for item in stats for key, entry in item.items() if 'sample_std' in entry]

    # Initialize lists for standard deviation values
    cluster_std_x, cluster_std_y, cluster_std_z = [], [], []
    sample_std_x, sample_std_y, sample_std_z = [], [], []

    # Extract standard deviation values for each axis (x, y, z)
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

    # Return results in a structured dictionary
    return {
        "num_points": num_points,
        "cluster_sizes": cluster_sizes,
        "cluster_std_x": cluster_std_x,
        "cluster_std_y": cluster_std_y,
        "cluster_std_z": cluster_std_z,
        "sample_std_x": sample_std_x,
        "sample_std_y": sample_std_y,
        "sample_std_z": sample_std_z
    }
