import torch
from torch.utils.data import Dataset as DS
import pandas as pd
import numpy as np
import os

from helpers.data import get_label, read_pts_file, assign_labels


class Dataset(DS):
    def __init__(self, root_folder,
                       suffix,
                       transform=None,
                       classes_to_use=None,
                       data_augmentation=False,
                       remove_part_prob=0.4,
                       remove_outliers=False,
                       remove_corners=False):
        """
        Initialize the dataset class.
        :param root_folder: path to main folder containing data
        :param suffix: suffix of the files to be added to the dataset
        :param transform: pre-defined transformations to be applied to the point cloud e.g. ToTensor or Padding
        :param classes_to_use: if there are multiple classes in the root_folder, by defining the parameter we can
        choose which ones to use. Use 'all' to get all the classes from root_folder.
        :param data_augmentation: boolean to decide if augmentation should be applied or not
        :param remove_part_prob: defines the probability of removing a point from the point cloud. If 0, no points are removed.
        :param remove_outliers: boolean to decide if outliers should be removed or not
        :param remove_corners: boolean to decide if corners should be removed or not
        """
        super().__init__()
        if classes_to_use is None:
            classes_to_use = ['cube', 'pyramid']
        self.root_folder = root_folder
        self.suffix = suffix
        self.data_augmentation = data_augmentation
        self.remove_outliers = remove_outliers
        if 'all' in classes_to_use:
            self.classes_to_use = [elem for elem in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, elem))]
        else:
            self.classes_to_use = classes_to_use
        self.transform = transform
        #if self.suffix == '.pts':
        self.labels = assign_labels(root_folder)
        self.p_remove_point = remove_part_prob
        self.remove_corners = remove_corners
        # Find all files in the root folder with the given suffix
        self.filepaths = []
        for subdir, dirs, files in os.walk(self.root_folder):
            for file in files:
                if file.endswith(self.suffix) and subdir.split('/')[-1] in self.classes_to_use:
                    self.filepaths.append(os.path.join(subdir, file))

    def _augment_data(self, point_cloud):
        # Define a random rotation matrix
        theta = np.random.uniform(0, 2 * np.pi)  # Rotation angle
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Apply the rotation to each point in the point cloud
        augmented_point_cloud = np.dot(point_cloud[:, :3], rotation_matrix.T)
        #augmented_point_cloud = np.column_stack((augmented_point_cloud, point_cloud[:, 3:]))  # Append the intensity back
        return augmented_point_cloud

    def _remove_points(self, point_cloud):
        partial_pc = point_cloud
        num_points = len(point_cloud)
        remove_points_mask = np.random.choice([True, False], num_points, p=[self.p_remove_point , 1 - self.p_remove_point ])
        partial_pc = point_cloud[~remove_points_mask]
        return partial_pc

    def _remove_outliers(self, point_cloud):
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=None, algorithm='best', alpha=0.7,metric='euclidean')
        cluster_labels = clusterer.fit_predict(point_cloud)
        from collections import Counter
        label_counts = Counter(cluster_labels)
        label_with_fewer_points = min(label_counts, key=label_counts.get)
        filtered_point_cloud = point_cloud[cluster_labels != label_with_fewer_points]
        return filtered_point_cloud

    def _remove_corners(self, point_cloud):
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=None, algorithm='best', alpha=0.7,metric='euclidean')
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
        else:
            partial_pc = self._remove_points(point_cloud)

        return partial_pc

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        filepath = self.filepaths[index]
        if 'csv' in self.suffix:
        # Load data from file
            arr = np.array(pd.read_csv(filepath))
        elif 'pts' in self.suffix:
            arr = read_pts_file(filepath)
        else:
            raise NotImplementedError("This data type is not implemented at the moment.")
        partial_arr = None
        #target = (self.highest_shape, arr.shape[-1])
        #padding = [(0, target[0] - arr.shape[0]), (0, 0)]
        #padded_arr = np.pad(arr, padding)
        # Process data into a tensor
        #tensor_data = torch.tensor(padded_arr, dtype=torch.float)

        # Save the tensor to a folder
        #save_folder = os.path.join(self.root_folder, 'processed_data')
        #if not os.path.exists(save_folder):
        #    os.makedirs(save_folder)
        #save_filepath = os.path.join(save_folder, f'{self.filename}_{index}.pt')
        #torch.save(tensor_data, save_filepath)

        cls = os.path.dirname(filepath).split('/')[-1]
        if cls in self.classes_to_use:
            label = self.labels[cls]
        else:
            raise NotImplementedError("This label is not included in the current dataset.")


        if self.data_augmentation:
            #theta = np.random.uniform(0, np.pi * 2)
            #rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            #arr[:, [0, 2]] = arr[:, [0, 2]].dot(rotation_matrix)  # random rotation
            #arr += np.random.normal(0, 0.02, size=arr.shape)  # random jitter
            arr = self._augment_data(arr)

        if self.p_remove_point > 0 and self.remove_corners is False:
            partial_arr = self._remove_points(arr)

        if self.remove_corners:
            partial_arr = self._remove_corners(arr)

        if self.remove_outliers:
            arr = self._remove_outliers(arr)
            partial_arr = self._remove_outliers(partial_arr)

        sample = {'pc': arr,
                  'partial_pc': partial_arr,
                  'label': np.array(label),
                  'path': filepath}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

