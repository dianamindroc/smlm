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


def find_top_n_smallest(arr, n):
    if n <= 0:
        return {"error": "n must be a positive integer"}

    if len(arr) < n:
        return {"error": f"Array has fewer than {n} elements"}

    # Create indexed array
    indexed_arr = [(value, index) for index, value in enumerate(arr)]

    # Sort by value
    indexed_arr.sort(key=lambda x: x[0])

    # Get top n smallest
    result = []
    for i in range(n):
        result.append({
            "value": indexed_arr[i][0],
            "index": indexed_arr[i][1]
        })

    return result


def find_top_n_largest(arr, n):
    if n <= 0:
        return {"error": "n must be a positive integer"}

    if len(arr) < n:
        return {"error": f"Array has fewer than {n} elements"}

    # Create indexed array
    indexed_arr = [(value, index) for index, value in enumerate(arr)]

    # Sort by value in descending order
    indexed_arr.sort(key=lambda x: x[0], reverse=True)

    # Get top n largest
    result = []
    for i in range(n):
        result.append({
            "value": indexed_arr[i][0],
            "index": indexed_arr[i][1]
        })

    return result


def find_best_errors_by_corner_group(data_dict):
    import pandas as pd
    import numpy as np
    """
    Find the best (lowest) L1 errors for each group of corners removed (0, 1, 2, etc.)

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing 'l1_errors', 'corner_labels', and optionally 'corner_names'

    Returns:
    --------
    dict
        Dictionary containing best errors and their information for each corner group
    """
    # Extract relevant data
    l1_errors = data_dict['l1_errors']
    corner_labels = data_dict['corner_labels']

    # Optional: extract corner names if available
    corner_names = data_dict.get('corner_names', None)

    # Create DataFrame for easier manipulation
    df_data = {
        'l1_error': l1_errors,
        'corner_count': corner_labels,
        'index': np.arange(len(l1_errors))  # Keep track of original indices
    }

    if corner_names is not None:
        df_data['corner_description'] = corner_names

    df = pd.DataFrame(df_data)

    # Find best (lowest) L1 error for each corner group
    best_indices = df.groupby('corner_count')['l1_error'].idxmin()
    best_errors = df.loc[best_indices]

    # Convert to dictionary with corner count as keys
    result = {}
    for corner_count, row in best_errors.iterrows():
        result[int(corner_count)] = {
            'l1_error': row['l1_error'],
            'index': int(row['index'])
        }
        if corner_names is not None:
            result[int(corner_count)]['description'] = row['corner_description']

    # Also include summary stats for each group
    group_stats = df.groupby('corner_count')['l1_error'].agg(['min', 'max', 'mean', 'std', 'count']).to_dict(
        orient='index')
    for corner_count, stats in group_stats.items():
        result[int(corner_count)]['stats'] = stats

    return result