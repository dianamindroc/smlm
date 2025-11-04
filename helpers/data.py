import os
import numpy as np
import torch.nn.functional as F
import pandas as pd

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


def get_highest_shape(root_folder, classes, subfolders=('iso', 'aniso'), suffix='.csv'):
    if 'all' in classes:
        classes_to_use = [elem for elem in os.listdir(root_folder) if
                               os.path.isdir(os.path.join(root_folder, elem))]
    else:
        classes_to_use = classes
    highest = 0
    for cls in classes_to_use:
        for subfolder in subfolders:
            folder = os.path.join(root_folder, cls, subfolder)
            if not os.path.isdir(folder):
                continue
            for file in os.listdir(folder):
                full_path = os.path.join(folder, file)
                if os.path.isfile(full_path) and file.endswith(suffix):
                    try:
                        with open(full_path, "r", encoding="utf-8") as fp:
                            line_count = sum(1 for _ in fp)
                        if line_count:
                            line_count -= 1  # discount header row
                        if line_count > highest:
                            highest = line_count
                    except Exception as e:
                        print(f"Failed to read {full_path}: {e}")
    return highest


def _remove_corners(point_cloud):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=None, algorithm='best', alpha=0.7,
                                metric='euclidean')
    cluster_labels = clusterer.fit_predict(point_cloud[['x', 'y', 'z']])
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


def add_anisotropy(point_cloud, anisotropy_factor, anisotropic_axis):
    if anisotropic_axis == 'x':
        axis = 0
    elif anisotropic_axis == 'y':
        axis = 1
    elif anisotropic_axis == 'z':
        axis = 2

    augmented_point_cloud = np.copy(point_cloud)
    augmented_point_cloud[:, axis] *= anisotropy_factor
    return augmented_point_cloud


def get_partial_data(root_folder, subfolder_list):
    import pandas as pd
    for append in subfolder_list:
        path = root_folder + append
        pcs = [os.path.join(path, entry) for entry in os.listdir(path) if entry.endswith('.csv')]
        pcs_array = [pd.read_csv(pc) for pc in pcs]
        #partial_pcs = [add_anisotropy(pc, 2, 'z') for pc in pcs_array]
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
