import glob
import os

import numpy as np
import torch.nn.functional as F
import pandas as pd

def add_folder_label_to_csv(csv_folder, folder1_name, folder2_name):
    for filename in os.listdir(csv_folder):
        file_path = os.path.join(csv_folder, filename)

        if os.path.isfile(file_path) and filename.endswith('.csv'):
            df = pd.read_csv(file_path)
            folder_name = os.path.basename(os.path.dirname(file_path))

            if folder_name == folder1_name:
                df['label'] = 0
            elif folder_name == folder2_name:
                df['label'] = 1

            labeled_file_path = os.path.join(csv_folder, f'labeled_{filename}')
            df.to_csv(labeled_file_path, index=False)

            print(f'Label added to {filename} and saved as labeled_{filename}.')


def get_label(name, classes):
    if classes[0] == name:
        label = 0
    else:
        label = 1
    return label


def assign_labels(root_folder):
    classes = {}
    class_label = 0

    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if os.path.isdir(folder_path):
            classes[folder] = class_label
            class_label += 1

    return classes


def pad_tensor(current_shape, target_shape):

    # Calculate the amount of padding needed along the second dimension
    padding = target_shape[0] - current_shape.size(0)

    # Pad the tensor with zeros along the second dimension
    padded_tensor = F.pad((0, 0, 0, padding, 0, 0), )

    return padded_tensor


def get_highest_shape(root_folder, classes):
    csv_file = os.path.join(os.path.dirname(os.path.abspath(os.curdir)),'resources', 'highest_shape.csv')
    #csv_file = os.path.join('/home/squirrel/coding/smlm', 'resources', 'highest_shape.csv')
    dataset_name = os.path.basename(root_folder)
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        try:
            highest = df.loc[df['dataset'] == dataset_name, 'highest_shape']
            return int(highest)
        except:
            print('Dataset not found in the CSV file.')
    else:
        print(f"CSV file '{csv_file}' does not exist.")

    highest = 0
    if 'all' in classes:
        cls = [elem for elem in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, elem))]
    else:
        cls = classes
    for cl in cls:
        folder = os.path.join(root_folder, cl)
        files = os.listdir(folder)
        for file in files:
            if len(pd.read_csv(os.path.join(folder, file))) > highest:
                highest = len(pd.read_csv(os.path.join(folder, file)))
    update_csv(csv_file, dataset_name, highest)
    return highest



def read_pts_file(file_path):
    with open(file_path, 'r') as file:
        # Skip the first two lines (header information)
        next(file)
        next(file)

        # Read the remaining lines and extract the coordinates
        points = []
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x, y, z])

    return np.array(points)


import pandas as pd

def update_csv(csv_file, dataset_name, highest_shape):
    data = {'dataset': [dataset_name],
            'highest_shape': [highest_shape]}

    df = pd.DataFrame(data)

    if not pd.api.types.is_string_dtype(df['dataset']):
        df['dataset'] = df['dataset'].astype(str)

    if not pd.api.types.is_numeric_dtype(df['highest_shape']):
        df['highest_shape'] = df['highest_shape'].astype(int)

    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        existing_df = pd.read_csv(csv_file)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(csv_file, index=False)

def _remove_corners(point_cloud):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=None, algorithm='best', alpha=0.7,
                                metric='euclidean')
    cluster_labels = clusterer.fit_predict(point_cloud)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    p_remove_cluster = 0.5

    if np.random.rand() < p_remove_cluster and n_clusters > 0:
        # Exclude noise label (-1) from the list of unique labels if it exists
        unique_clusters = set(cluster_labels)
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        unique_clusters = list(unique_clusters)

        # Randomly select a cluster to remove
        cluster_to_remove = np.random.choice(unique_clusters)

        # Create a mask for points to keep
        mask = cluster_labels != cluster_to_remove

        # Apply the mask to keep only the points that are not in the selected cluster
        partial_pc = point_cloud[mask]
        return partial_pc
    else:
        return point_cloud


def get_partial_data(root_folder, subfolder_list):
    import pandas as pd
    for append in subfolder_list:
        path = root_folder + append
        pcs = [os.path.join(path, entry) for entry in os.listdir(path) if entry.endswith('.csv')]
        pcs_array = [pd.read_csv(pc, index_col=0) for pc in pcs]
        partial_pcs = [_remove_corners(pc) for pc in pcs_array]
        for i, pc in enumerate(partial_pcs):
            pathh = root_folder.split('gt')[0] + 'partial/' + append + '/smlm_sample'
            if 'test' in root_folder:
                pathh = root_folder + append + '/smlm_sample'
            print('Partial path is: ', pathh)
            pc.to_csv(pathh + str(i) + '.csv', index=False)


def preprocess_all_data(root_folder, subfolder_list, padding_length):
    import h5py
    import pandas as pd
    import numpy as np
    for append in subfolder_list:
        path = root_folder + append
        pcs = [os.path.join(path, entry) for entry in os.listdir(path) if entry.endswith('.csv')]
        for i, pc in enumerate(pcs):
            df = pd.read_csv(pc)
            if len(df.columns) > 3:
                df = pd.read_csv(pc, index_col=0)
            structured_array = np.array(df)
            num_points_needed = padding_length - structured_array.shape[0]
            if num_points_needed > 0:
                # Select random indices from the point cloud to duplicate
                indices_to_duplicate = np.random.choice(structured_array.shape[0], num_points_needed, replace=True)
                duplicated_points = structured_array[indices_to_duplicate]

                # Concatenate the original point cloud with the duplicated points
                padded_point_cloud = np.vstack((structured_array, duplicated_points))
            else:
                padded_point_cloud = structured_array

            # Create a mask indicating original points with ones and duplicated points with zeros

            pathh = root_folder + append + '/smlm_sample'
            with h5py.File(pathh + str(i) + '_hdf.h5', 'w') as file:
                # Create a dataset directly under the root of the HDF5 file
                file.create_dataset('data', data=padded_point_cloud)


def get_files_list(root_folder):
    # Specify the path for the output text file
    output_file_path = os.path.dirname(root_folder) + '/filenames.txt'

    parent_folder_name = os.path.basename(root_folder)

    with open(output_file_path, 'w') as file:
        # List all files in the directory
        for filename in os.listdir(root_folder):
            # Check if the file has the .h5 extension
            if filename.endswith('.h5'):
                # Remove the file extension
                filename_without_suffix, _ = os.path.splitext(filename)
                # Append parent folder name to filename
                full_filename = f"{parent_folder_name}/{filename_without_suffix}"
                # Write each full filename to the text file, followed by a newline
                file.write(full_filename + '\n')


def get_max_length(root_folder):
    # Initialize the maximum length variable
    max_length = 0
    # Loop through each sub-folder in the root directory
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):  # Check if it's a directory
            # Loop through each file in the sub-folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.h5'):
                    file_path = os.path.join(folder_path, file_name)
                    # Read the CSV file
                    df = pd.read_hdf(file_path)
                    # Update the maximum length if the current one is larger
                    if len(df) > max_length:
                        max_length = len(df)
    return max_length


