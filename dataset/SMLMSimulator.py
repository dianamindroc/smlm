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

        if self.structure_type == 'box':
            self.structure_ground_truth = self.get_corners_3d_box()
        elif self.structure_type == 'tetrahedron':
            self.structure_ground_truth = self.get_corners_3d_tetrahedron()
        else:
            raise ValueError(f"Unsupported structure_type: {self.structure_type}")

        # Pre-generate seeds so each index yields deterministic samples without storing them.
        rng = np.random.default_rng()
        self._sample_seeds = rng.integers(low=0, high=np.iinfo(np.int64).max, size=self.num_samples, dtype=np.int64)

    def _simulate_point_cloud(self, rng):
        """Simulate a single point cloud using vectorized noise generation."""
        simulated_points = []
        for corner in self.structure_ground_truth:
            intensity = rng.uniform(*self.dye_properties['intensity_range'])
            precision = rng.uniform(*self.dye_properties['precision_range'])
            blinking_times = rng.integers(*self.dye_properties['blinking_times_range'])
            simulated_points.append(self.simulate_blinks(corner, blinking_times, intensity, precision, rng))
        return np.concatenate(simulated_points, axis=0)

    def _augment_data(self, point_cloud, rng):
        # Define a random rotation matrix
        theta = rng.uniform(0, 2 * np.pi)  # Rotation angle
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])

        # Apply the rotation to each point in the point cloud
        augmented_point_cloud = np.dot(point_cloud[:, :3], rotation_matrix.T)
        rotated_ground_truth = np.dot(self.structure_ground_truth, rotation_matrix.T)
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

            if rng.random() < p_remove_cluster:
                # Randomly select a cluster to remove
                cluster_to_remove = rng.integers(0, n_clusters)

                # Create a mask for points to keep
                mask = kmeans.labels_ != cluster_to_remove

                # Apply the mask to keep only the points that are not in the selected cluster
                pc_without_corners = pc[mask]

        if self.remove_points:
            # Probability of removing a point
            p_remove_point = 0.05
            num_points = len(pc)
            remove_points = rng.random(num_points) < p_remove_point
            pc = pc[~remove_points]

        return pc, pc_without_corners, rotated_ground_truth

    # Define the function to simulate blinks with Gaussian noise
    @staticmethod
    def simulate_blinks(point, blinking_times, intensity, baseline_precision, rng):
        noise = rng.normal(0, baseline_precision, size=(blinking_times, 3))
        return point + noise

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
        return self.num_samples

    def __getitem__(self, idx):
        rng = np.random.default_rng(self._sample_seeds[idx])
        point_cloud = self._simulate_point_cloud(rng)
        pc_without_corners = None
        ground_truth = self.structure_ground_truth.copy()

        if self.augment:
            point_cloud, pc_without_corners, ground_truth = self._augment_data(point_cloud, rng)

        label = self.structure_type
        partial = pc_without_corners if pc_without_corners is not None else point_cloud

        sample = {
            'pc': point_cloud,
            'pc_anisotropic': point_cloud,
            'partial_pc': partial,
            'ground_truth': ground_truth,
            'label': label,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
