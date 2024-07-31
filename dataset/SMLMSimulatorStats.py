import numpy as np
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import pandas as pd

class DNAOrigamiSimulator:
    def __init__(self, num_samples, structure_type, dye_properties, stats, anisotropy_factor, transform=None):
        """
        :param num_samples: Number of samples to generate
        :param structure_type: Type of the structure ('box' or 'tetrahedron')
        :param dye_properties: Dictionary containing 'intensity_range', 'precision_range', 'blinking_times_range'
        :param point_dist_params: Tuple (mu, sigma) for the number of points distribution
        :param std_devs: Tuple (std_x, std_y, std_z) for the standard deviations in each axis
        :param transform: Optional transformation to apply to the point cloud
        """
        self.num_samples = num_samples
        self.structure_type = structure_type
        self.dye_properties = dye_properties
        self.stats = stats
        self.transform = transform
        self.data = self._simulate_datasets(anisotropy_factor)

    def get_corners_3d_box(self, length=30, width=30, height=30):
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

    @staticmethod
    def get_corners_3d_tetrahedron(mean_side_length=1.5, std_dev=0.1, min_length=0.5, max_length=2):
        """
        Generate corners of a tetrahedron with side length sampled from a distribution.

        :param mean_side_length: Mean of the side length distribution
        :param std_dev: Standard deviation of the side length distribution
        :param min_length: Minimum allowed side length
        :param max_length: Maximum allowed side length
        :return: numpy array of shape (4, 3) representing the corners of the tetrahedron
        """
        # Sample the side length from a normal distribution
        side_length = np.clip(np.random.normal(mean_side_length, std_dev), min_length, max_length)

        # Calculate tetrahedron properties
        height = np.sqrt(2) / np.sqrt(3) * side_length
        circumradius = np.sqrt(3) / 4 * side_length

        # Define the corners
        corners = np.array([
            [0, 0, -height / 3],
            [circumradius, 0, height * 2 / 3],
            [-circumradius / 2, circumradius * np.sqrt(3) / 2, height * 2 / 3],
            [-circumradius / 2, -circumradius * np.sqrt(3) / 2, height * 2 / 3]
        ])

        return corners

    def rotate_model(self):
        """Applies a random rotation to the ground truth structure."""
        random_rotation = R.random()
        rotated_model = random_rotation.apply(self.structure_ground_truth)
        self.structure_ground_truth = [tuple(corner) for corner in rotated_model]

    def simulate_blinks(self, point, blinking_times, anisotropy_factor, baseline_precision, std_x, std_y, std_z):
        """Simulates the blinking points with added noise."""
        blinks = []
        for _ in range(blinking_times):
            base_noise = np.random.normal(0, baseline_precision, 3)

            # Adding the std deviations to simulate the spread in each dimension
            anisotropic_noise = np.array([np.random.normal(0, std_x),
                                          np.random.normal(0, std_y),
                                          np.random.normal(0, std_z * anisotropy_factor)
                                          ])

            noisy_point = point + base_noise + anisotropic_noise
            blinks.append(noisy_point)
        return blinks

    def _simulate_datasets(self, anisotropy_factor):
        dataset = []
        if self.structure_type == 'box':
            self.structure_ground_truth = self.get_corners_3d_box()
        elif self.structure_type == 'tetrahedron':
            self.structure_ground_truth = self.get_corners_3d_tetrahedron()

        num_points_list = [stats['num_points'] for stats in self.stats]
        # Calculate mean (mu)
        mu = np.mean(num_points_list)
        # Calculate standard deviation (sigma)
        sigma = np.std(num_points_list)
        num_points = int(np.random.normal(mu, sigma))
        # threequarter = int(3/4*len(num_points))
        # new_loc = int(num_points + random.uniform(-percentage, percentage))
        std_dev_df = pd.DataFrame([stats['std_dev'] for stats in self.stats], columns=['x', 'y', 'z'])
        overall_std_dev = std_dev_df.mean()
        std_x, std_y, std_z = overall_std_dev['x'], overall_std_dev['y'], overall_std_dev['z']

        for _ in range(self.num_samples):
            # Rotate the model before generating points
            self.rotate_model()

            point_cloud = []
            num_points = int(np.random.normal(mu, sigma))
            total_points_generated = 0

            while total_points_generated < num_points:
                for corner in self.structure_ground_truth:
                    if total_points_generated >= num_points:
                        break

                    #intensity = np.random.uniform(*self.dye_properties['intensity_range'])
                    precision = np.random.uniform(*self.dye_properties['precision_range'])
                    blinking_times = np.random.randint(*self.dye_properties['blinking_times_range'])

                    # Generate blinks for the current corner point
                    blinks = self.simulate_blinks(corner, blinking_times, anisotropy_factor, precision, std_x, std_y, std_z)
                    point_cloud.extend(blinks)

                    total_points_generated += len(blinks)

                    if total_points_generated >= num_points:
                        break

            # If we have more points than needed, trim the excess
            if len(point_cloud) > num_points:
                point_cloud = point_cloud[:num_points]

            dataset.append((self.structure_ground_truth, np.array(point_cloud)))
        return dataset

    def save_to_csv(self, file_prefix='sample'):
        """Saves the generated dataset to CSV files."""
        for idx, (ground_truth, point_cloud) in enumerate(self.data):
            df = pd.DataFrame(point_cloud, columns=['x', 'y', 'z'])
            df.to_csv(f'{file_prefix}_{idx}.csv', index=False)

    def get_sample(self, idx):
        """Returns a single sample from the dataset."""
        ground_truth, point_cloud = self.data[idx]
        if self.transform is not None:
            point_cloud = self.transform(point_cloud)

        label = self.structure_type
        return {'pc': point_cloud, 'label': label}
