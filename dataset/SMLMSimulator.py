import numpy as np
from torch.utils.data import Dataset
from sklearn.cluster import KMeans


# Define the DNAOrigamiSimulator class
class DNAOrigamiSimulator(Dataset):
    def __init__(self, num_samples, structure_type, dye_properties,
                 augment=False,
                 remove_corners=False,
                 remove_points=False,
                 transform=None):
        self.num_samples = num_samples
        self.structure_type = structure_type
        self.dye_properties = dye_properties
        self.augment = augment
        self.remove_corners = remove_corners
        self.remove_points = remove_points
        self.transform = transform
        self.data = self._simulate_datasets()

    def _simulate_datasets(self):
        dataset = []
        if self.structure_type == 'box':
            self.structure_ground_truth = self.get_corners_3d_box()
        elif self.structure_type == 'tetrahedron':
            self.structure_ground_truth = self.get_corners_3d_tetrahedron()
        for _ in range(self.num_samples):
            point_cloud = []
            for corner in self.structure_ground_truth:
                intensity = np.random.uniform(*self.dye_properties['intensity_range'])
                precision = np.random.uniform(*self.dye_properties['precision_range'])
                blinking_times = np.random.randint(*self.dye_properties['blinking_times_range'])
                point_cloud.extend(self.simulate_blinks(corner, blinking_times, intensity, precision))
            dataset.append((self.structure_ground_truth, np.array(point_cloud)))
        return dataset

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
        pc = augmented_point_cloud

        pc_without_corners = pc

        if self.remove_corners:
            # Probability of removing a corner
            if self.structure_type == 'box':
                n_clusters = 8  # Adjust based on your actual number of corners
            else:
                n_clusters = 4

            # Perform K-Means Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pc)

            # Probability of removing a cluster
            p_remove_cluster = 0.5

            if np.random.rand() < p_remove_cluster:
                # Randomly select a cluster to remove
                cluster_to_remove = np.random.choice(n_clusters)

                # Create a mask for points to keep
                mask = kmeans.labels_ != cluster_to_remove

                # Apply the mask to keep only the points that are not in the selected cluster
                pc_without_corners = pc[mask]

        # Do not use this yet
        if self.remove_points:
            # Probability of removing a point
            p_remove_point = 0.05
            num_points = len(pc)
            remove_points = np.random.choice([True, False], num_points, p=[p_remove_point, 1 - p_remove_point])
            point_cloud = pc[~remove_points]
            pc = point_cloud

        return pc, pc_without_corners

    # Define the function to simulate blinks with Gaussian noise
    @staticmethod
    def simulate_blinks(point, blinking_times, intensity, baseline_precision):
        blinks = []
        for _ in range(blinking_times):
            intensity = intensity
            noise = np.random.normal(0, baseline_precision, 3)
            noisy_point = point + noise
            # blinks.append(np.append(noisy_point, intensity))
            blinks.append(noisy_point)
        return blinks

    # Define the function to get the corners of a 3D box
    @staticmethod
    def get_corners_3d_box(length=30, width=30, height=30):
        half_length = length / 2
        half_width = width / 2
        half_height = height / 2
        corners = [
            [-half_length, -half_width, -half_height],
            [half_length, -half_width, -half_height],
            [half_length, half_width, -half_height],
            [-half_length, half_width, -half_height],
            [-half_length, -half_width, half_height],
            [half_length, -half_width, half_height],
            [half_length, half_width, half_height],
            [-half_length, half_width, half_height]
        ]
        return np.array(corners)

    # Function to create the corners of a tetrahedron
    @staticmethod
    def get_corners_3d_tetrahedron(side_length=100):
        height = np.sqrt(2) / np.sqrt(3) * side_length
        circumradius = np.sqrt(3) / 4 * side_length
        corners = [
            [0, 0, -height / 3],
            [circumradius, 0, height * 2 / 3],
            [-circumradius / 2, circumradius * np.sqrt(3) / 2, height * 2 / 3],
            [-circumradius / 2, -circumradius * np.sqrt(3) / 2, height * 2 / 3]
        ]
        return np.array(corners)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ground_truth, point_cloud = self.data[idx]
        pc_without_corners = None
        if self.augment:
            point_cloud, pc_without_corners = self._augment_data(point_cloud)
        if self.transform is not None:
            point_cloud = self.transform(point_cloud)
            if pc_without_corners is not None:
                pc_without_corners = self.transform(pc_without_corners)

        label = self.structure_type

        sample = {'pc': point_cloud,
                  'partial_pc': pc_without_corners,
                  'label': label}

        return sample
