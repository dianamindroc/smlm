import torch
from torch.utils.data import Dataset as DS
import pandas as pd
import numpy as np
import os
import open3d as o3d
from helpers.data import assign_labels
from helpers.readers import read_pts_file


class PairedAnisoIsoDataset(DS):
    def __init__(self, root_folder,
                 suffix,
                 transform=None,
                 classes_to_use=None,
                 remove_part_prob=0.0,
                 remove_outliers=False,
                 remove_corners=False,
                 number_corners_to_remove=1):
        """
        Dataset that returns both aniso and iso versions of each point cloud sample.

        :param root_folder: Root directory with class folders containing 'aniso' and 'iso' subfolders.
        :param suffix: File extension to look for (e.g. '.csv' or '.pts').
        :param transform: Optional transform applied to the sample dict.
        :param classes_to_use: List of class names to include or 'all'.
        :param remove_part_prob: Dropout probability for simulating incomplete inputs.
        :param remove_outliers: Whether to apply statistical outlier removal.
        :param remove_corners: Whether to remove HDBSCAN clusters.
        :param number_corners_to_remove: Int or list, how many clusters to remove.
        """
        super().__init__()
        self.root_folder = root_folder
        self.suffix = suffix
        self.transform = transform
        self.p_remove_point = remove_part_prob
        self.remove_outliers = remove_outliers
        self.remove_corners = remove_corners
        self.number_corners_to_remove = number_corners_to_remove

        if 'all' in classes_to_use:
            self.classes_to_use = [elem for elem in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, elem))]
        else:
            self.classes_to_use = classes_to_use

        self.labels = assign_labels(root_folder)
        self.paired_filepaths = []
        self.available_classes = self.classes_to_use

        for cls in self.labels.keys():  # Use keys from label dict (i.e. tetra, cube)
            class_dir = os.path.join(root_folder, cls)
            iso_dir = os.path.join(class_dir, 'iso')
            aniso_dir = os.path.join(class_dir, 'aniso')

            if not os.path.isdir(iso_dir) or not os.path.isdir(aniso_dir):
                continue  # skip classes without full iso/aniso folders

            for file in os.listdir(iso_dir):
                if file.endswith(self.suffix):
                    iso_path = os.path.join(iso_dir, file)
                    aniso_path = os.path.join(aniso_dir, file)
                    if os.path.isfile(aniso_path):  # only include paired samples
                        self.paired_filepaths.append({
                            'class': cls,
                            'filename': file,
                            'iso_path': iso_path,
                            'aniso_path': aniso_path
                        })

        if len(self.paired_filepaths) == 0:
            print(f"Warning: No paired samples found in {root_folder}. Check folder structure and suffix.")

    @staticmethod
    def _remove_statistical_outliers(point_cloud, nb_neighbors=40, std_ratio=1.0):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        return point_cloud[np.asarray(ind), :]

    def _remove_points(self, point_cloud):
        mask = np.random.rand(len(point_cloud)) > self.p_remove_point
        return point_cloud[mask]

    def _remove_corners(self, point_cloud, number_to_remove):
        import hdbscan
        if isinstance(number_to_remove, list):
            number_to_remove = np.random.choice(number_to_remove)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=20, alpha=0.7)
        cluster_labels = clusterer.fit_predict(point_cloud)
        unique_clusters = list(set(cluster_labels) - {-1})

        if not unique_clusters:
            return point_cloud, 0

        clusters_to_remove = np.random.choice(unique_clusters,
                                              size=min(number_to_remove, len(unique_clusters)),
                                              replace=False)
        mask = ~np.isin(cluster_labels, clusters_to_remove)
        return point_cloud[mask], len(clusters_to_remove)

    def __len__(self):
        return len(self.paired_filepaths)

    def __getitem__(self, index):
        item = self.paired_filepaths[index]
        cls = item['class']
        filename = item['filename']
        iso_path = item['iso_path']
        aniso_path = item['aniso_path']

        def load_file(filepath):
            if self.suffix == '.csv':
                data = pd.read_csv(filepath)[['x', 'y', 'z']]
                return data.values[:, :3]
            elif self.suffix == '.pts':
                return read_pts_file(filepath)[:, :3]
            else:
                raise NotImplementedError("Unsupported file format")

        iso_pc = load_file(iso_path)
        aniso_pc = load_file(aniso_path)

        if self.remove_outliers:
            iso_pc = self._remove_statistical_outliers(iso_pc)
            aniso_pc = self._remove_statistical_outliers(aniso_pc)

        # Simulate dropout / occlusion (only on input side, i.e. aniso)
        if self.remove_corners:
            partial_pc, corner_count = self._remove_corners(aniso_pc, self.number_corners_to_remove)
        elif self.p_remove_point > 0:
            partial_pc = self._remove_points(aniso_pc)
            corner_count = 0
        else:
            partial_pc = aniso_pc
            corner_count = 0

        corner_label_name = {
            0: 'No Corners Removed',
            1: 'One Corner Removed',
            2: 'Two Corners Removed'
        }.get(corner_count, f'{corner_count} Corners Removed')

        label = self.labels.get(cls, None)

        sample = {
            'pc': iso_pc,
            'pc_anisotropic': aniso_pc,
            'partial_pc': partial_pc,
            'label': np.array(self.labels[cls]),
            'label_name': cls,
            'iso_path': iso_path,
            'aniso_path': aniso_path,
            'filename': filename,
            'corner_label': np.array(corner_count),
            'corner_label_name': corner_label_name
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
