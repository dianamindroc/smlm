import torch
from torch.utils.data import Dataset as DS
import pandas as pd
import numpy as np
import os
import open3d as o3d
from helpers.data import get_label, assign_labels
from helpers.readers import read_pts_file


class Dataset(DS):
    def __init__(self, root_folder,
                       suffix,
                       transform=None,
                       classes_to_use=None,
                       data_augmentation=False,
                       remove_part_prob=0.4,
                       remove_outliers=False,
                       remove_corners=False,
                       number_corners_to_remove=1,
                       anisotropy=False,
                       anisotropy_axis='z',
                       anisotropy_factor=2.0):
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
        :param anisotropy: boolean to decide if anisotropy should be applied or not
        :param anisotropy_axis: defines the axis of anisotropy
        :param anisotropy_factor: defines the anisotropy factor
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
        self.number_corners_to_remove = number_corners_to_remove
        self.anisotropy = anisotropy
        self.anisotropy_axis = anisotropy_axis
        self.anisotropy_factor = anisotropy_factor
        # Find all files in the root folder with the given suffix
        self.filepaths = []
        for subdir, dirs, files in os.walk(self.root_folder):
            for file in files:
                if file.endswith(self.suffix) and subdir.split('/')[-1] in self.classes_to_use:
                    self.filepaths.append(os.path.join(subdir, file))

    @staticmethod
    def _rotate_data(point_cloud):
        # Define a random rotation matrix
        theta_x = np.random.uniform(0, 2 * np.pi)  # Rotation angle around x-axis
        theta_y = np.random.uniform(0, 2 * np.pi)  # Rotation angle around y-axis
        theta_z = np.random.uniform(0, 2 * np.pi)  # Rotation angle around z-axis

        # Rotation matrix around x-axis
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])

        # Rotation matrix around y-axis
        rot_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])

        # Rotation matrix around z-axis
        rot_z = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        rotation_matrix = rot_z @ rot_y @ rot_x

        # Apply the rotation to each point in the point cloud
        augmented_point_cloud = np.dot(point_cloud, rotation_matrix.T)

        #augmented_point_cloud = np.column_stack((augmented_point_cloud, point_cloud[:, 3:]))  # Append the intensity back
        return augmented_point_cloud

    @staticmethod
    def _add_anisotropy(point_cloud, anisotropy_factor, anisotropic_axis):
        if anisotropic_axis == 'x':
            axis = 0
        elif anisotropic_axis == 'y':
            axis = 1
        elif anisotropic_axis == 'z':
            axis = 2

        augmented_point_cloud = np.copy(point_cloud)

        # Generate and add noise to the anisotropic axis
        noise = np.random.normal(0, anisotropy_factor, size=point_cloud.shape[0])
        augmented_point_cloud[:, axis] += noise

        return augmented_point_cloud

    def _remove_points(self, point_cloud):
        #Remove random points from the point cloud - replicate underlabeling
        num_points = len(point_cloud)
        remove_points_mask = np.random.choice([True, False], num_points, p=[self.p_remove_point , 1 - self.p_remove_point ])
        partial_pc = point_cloud[~remove_points_mask]
        return partial_pc

    @staticmethod
    def _remove_outliers(point_cloud):
        # Remove outliers from the point cloud. Not used
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=None, algorithm='best', alpha=0.7, metric='euclidean')
        cluster_labels = clusterer.fit_predict(point_cloud)
        from collections import Counter
        label_counts = Counter(cluster_labels)
        label_with_fewer_points = min(label_counts, key=label_counts.get)
        filtered_point_cloud = point_cloud[cluster_labels != label_with_fewer_points]
        return filtered_point_cloud

    @staticmethod
    def _remove_statistical_outliers(point_cloud, nb_neighbors=40, std_ratio=1.0):
        spatial_data = point_cloud[:, :3]
        # Convert to Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(spatial_data)
        # Remove outliers
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        inlier_indices = np.asarray(ind)
        # Extract inlier rows from the original point cloud, keeping all columns
        inlier_cloud = point_cloud[inlier_indices, :]
        return inlier_cloud

    def _remove_corners(self, point_cloud, number_corners_to_remove):
        import hdbscan
        import numpy as np

        # Check if number_corners_to_remove is a single integer or a list
        if isinstance(number_corners_to_remove, int):
            # If it's a single integer, use it directly
            number_to_remove = number_corners_to_remove
        elif isinstance(number_corners_to_remove, list):
            # If it's a list, randomly choose one value from the list
            number_to_remove = np.random.choice(number_corners_to_remove)
        else:
            raise ValueError("number_corners_to_remove must be either an integer or a list of integers.")

        # Cluster the point cloud
        clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=None, algorithm='best', alpha=0.7,
                                    metric='euclidean')
        cluster_labels = clusterer.fit_predict(point_cloud)
        n_clusters = len(set(cluster_labels)) - (
            1 if -1 in cluster_labels else 0)  # Exclude noise (-1) from cluster count

        # Ensure we don't try to remove more clusters than exist
        clusters_to_remove = min(number_to_remove, n_clusters)

        if n_clusters > 0:
            # Exclude noise label (-1) from the list of unique labels if it exists
            unique_clusters = set(cluster_labels)
            if -1 in unique_clusters:
                unique_clusters.remove(-1)
            unique_clusters = list(unique_clusters)

            # Randomly select the clusters to remove
            clusters_to_remove = np.random.choice(unique_clusters, size=clusters_to_remove, replace=False)

            # Create a mask to keep points that are not in the selected clusters
            mask = np.isin(cluster_labels, clusters_to_remove, invert=True)

            # Apply the mask to keep only the points that are not in the selected clusters
            partial_pc = point_cloud[mask]
        else:
            partial_pc = point_cloud

        return partial_pc, number_to_remove

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

        cls = os.path.dirname(filepath).split('/')[-1]
        if cls in self.classes_to_use:
            label = self.labels[cls]
        else:
            raise NotImplementedError("This label is not included in the current dataset.")

        if self.remove_outliers:
            arr = self._remove_statistical_outliers(arr)

        if self.anisotropy:
            if 'exp' in filepath:
                arr_anisotropy = arr
            else:
                arr_anisotropy = self._add_anisotropy(arr, self.anisotropy_factor, self.anisotropy_axis)
        else:
            arr_anisotropy = arr

        if self.p_remove_point > 0 and self.remove_corners is False:
            partial_arr = self._remove_points(arr)

        if self.remove_corners:
            if 'exp' in filepath:
                partial_arr, corner = arr, 0
            if self.anisotropy:
                partial_arr, corner = self._remove_corners(arr_anisotropy, self.number_corners_to_remove)
            else:
                partial_arr, corner = self._remove_corners(arr, self.number_corners_to_remove)
        else:
            partial_arr, corner = arr, 0

        corner_to_label_name = {0: 'No Corners Removed', 1: 'One Corner Removed', 2: 'Two Corners Removed'}

        sample = {'pc': arr,
                  'pc_anisotropic': arr_anisotropy,
                  'partial_pc': partial_arr,
                  'label': np.array(label),
                  'path': filepath,
                  'label_name': cls,
                  'corner_label': np.array(corner),
                  'corner_label_name': corner_to_label_name[corner]}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

