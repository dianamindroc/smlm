import matplotlib.pyplot as plt
import matplotlib
import math
import pandas as pd
import numpy as np
import os
from scipy.stats import lognorm
from scipy.spatial.transform import Rotation as R
from helpers.pc_stats import load_and_analyze_point_clouds
from helpers.misc import extract_stats
from scipy.stats import ks_2samp
import datetime

class SMLMDnaOrigami:
    """
    Class to simulate a DNA origami
    :param struct_type: string describing the type of structures. Choose from box, pyramid, tetrahedron, sphere.
    :param number_dna_origami_samples: Number of DNA origami samples to simulate
    :param stats: Dictionary of statistics about the DNA origami to use during simulation
    :param apply_rotation: Boolean to apply the rotation to the DNA origami model structure
    """

    def __init__(self, struct_type: str, number_dna_origami_samples: int, stats=None, apply_rotation=True, seed=1234, side=1, root_folder='/home/dim26fa/data'):
        np.random.seed(seed)
        self.current_rotation = None
        self.struct_type = struct_type
        self.number_samples = number_dna_origami_samples
        self.dna_origami_list = []
        self.dna_origami = []
        self.apply_rotation = apply_rotation
        if stats is not None:
            self.stats = stats
            print('You provided a stats file. Simulating with prior knowledge.')
            self.fitted_distributions = self.fit_distributions()
        else:
            self.stats = None
            print('You did not provide a stats file. Simulating without prior knowledge.')
        if struct_type == 'cube':
            x, y, z, size = [int(x) for x in input("Enter cube coordinates and maximum size (separate by space): ").split()]
            self.model_structure = self.generate_cube(x, y, z, size)
        elif struct_type == 'pyramid':
            base, height = [int(x) for x in input("Enter pyramid base edge size and height (separate by space): ").split()]
            self.model_structure = self.generate_pyramid(base, height)
        elif struct_type == 'tetrahedron':
            #self.side = int(input("Enter tetrahedron side length: "))
            self.side = side
            #self.k = float(input("Enter scaling factor for std_dev (relative to side length): "))
            self.model_structure = self.generate_tetrahedron(self.side)
        elif struct_type == 'sphere':
            radius, latitude_divisions, longitude_divisions = [int(x) for x in input("Enter sphere radius, "
                                                                                     "nr. of latitude_divisions "
                                                                                     "and nr. of longitude_divisions: ").split()]
            self.model_structure = self.generate_sphere(radius, latitude_divisions, longitude_divisions)
        else:
            raise NotImplementedError("Only cube and pyramid supported at the moment :)")
        #self.radius = int(input("What is the maximum radius in which to generate SMLM samples?"))
        #self.number_localizations, self.percentage = [int(x) for x in input("How many localizations to generate around each center and with which variation (separate by space)?").split()]
        #self.uncertainty_factor = int(input("What is the uncertainty factor for the localizations?"))
        #self.base_folder = input("Where to save the generated samples?")
        date = datetime.datetime.now().date().strftime("%d%m")
        self.base_folder = os.path.join(root_folder, f'sim_dna_old_{date}_seed{seed}')
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
        print('Going to generate ' + str(number_dna_origami_samples) + ' samples for ' + struct_type + '.')
        #self.generate_all_dna_origami_smlm_samples()

    def simulate_dna_origami(self):
        return self.generate_all_dna_origami_smlm_samples()


    def fit_distributions(self):
        #shape_r, loc_r, scale_r = lognorm.fit(r_experimental)
        #median_r = np.median(r_experimental)
        #scale_r_corrected = median_r / np.exp(shape_r ** 2 / 2)
        cluster_std_df = pd.DataFrame([points['cluster_std'] for statss in self.stats for points in statss.values()],
                                      columns=['x', 'y', 'z'])
        # Fit lognormal distributions to experimental std_x, std_y, std_z
        shape_x, loc_x, scale_x_corrected = lognorm.fit(cluster_std_df['x'], floc=0)
        shape_y, loc_y, scale_y_corrected = lognorm.fit(cluster_std_df['y'], floc=0)
        shape_z, loc_z, scale_z_corrected = lognorm.fit(cluster_std_df['z'], floc=0)
        min_std_x, max_std_x = np.min(cluster_std_df['x']), np.max(cluster_std_df['x'])
        min_std_y, max_std_y = np.min(cluster_std_df['y']), np.max(cluster_std_df['y'])
        min_std_z, max_std_z = np.min(cluster_std_df['z']), np.max(cluster_std_df['z'])


        cluster_sizes = [pointsss['cluster_size']
                         for statss in self.stats
                         for pointsss in statss.values()
                         if 'cluster_size' in pointsss]
        min_sizes, max_sizes = np.min(cluster_sizes), np.max(cluster_sizes)
        cluster_sizesdf = np.array(cluster_sizes)
        shape, loc, scale = lognorm.fit(cluster_sizesdf, floc=0)

        # combined stds for simulating isotropic point cloud
        all_stds = np.concatenate([
            cluster_std_df['x'].values,
            cluster_std_df['y'].values,
            cluster_std_df['z'].values
        ])

        all_stds = all_stds[all_stds > 0]

        shape_comb, loc_comb, scale_comb = lognorm.fit(all_stds, floc=0)

        return {
            'cluster_std_distribution': {
                'x': {'shape': shape_x, 'loc': loc_x, 'scale': scale_x_corrected},
                'y': {'shape': shape_y, 'loc': loc_y, 'scale': scale_y_corrected},
                'z': {'shape': shape_z, 'loc': loc_z, 'scale': scale_z_corrected},
                'min_std_x': min_std_x,
                'max_std_x': max_std_x,
                'min_std_y': min_std_y,
                'max_std_y': max_std_y,
                'min_std_z': min_std_z,
                'max_std_z': max_std_z
            },
            'cluster_size_distribution': {
                'shape': shape,
                'loc': loc,
                'scale': scale,
                'min_size': min_sizes,
                'max_size': max_sizes
            },
            'cluster_std_combined': {
                'shape': shape_comb,
                'loc': loc_comb,
                'scale': scale_comb
            }
        }

    @staticmethod
    def generate_cube(x: int, y: int, z: int, size: int):
        """"
        Generates a cube with the first corner at the x,y,z coordinates and a specific size
        :param x: x coordinate
        :param y: y coordinate
        :param z: z coordinate
        :param size: the size of one edge
        :return: returns a list of tuples containing the coordinates
        """
        #size = random.choice(range(1, size))
        corners = [(x, y, z),
                   (x+size, y, z),
                   (x+size, y+size, z),
                   (x, y+size, z),
                   (x, y, z+size),
                   (x+size, y, z+size),
                   (x+size, y+size, z+size),
                   (x, y+size, z+size)]
        return corners

    @staticmethod
    def generate_pyramid(base_side_length, height):
        half_base = base_side_length / 2.0
        corners = []
        corners.append((0, height, 0))
        corners.append((-half_base, 0, -half_base))
        corners.append((half_base, 0, -half_base))
        corners.append((half_base, 0, half_base))
        corners.append((-half_base, 0, half_base))
        return corners

    @staticmethod
    def generate_tetrahedron(side_length):
        """
        Generates a regular tetrahedron with the given side length.
        The tetrahedron is centered at the origin.
        :param side_length: The length of each edge of the tetrahedron.
        :return: List of tuples containing the coordinates of the vertices.
        """
        # Calculate the height from the center of the base to the apex
        height = np.sqrt(2 / 3) * side_length

        # The radius of the circumscribed circle around the base triangle
        base_radius = side_length / np.sqrt(3)

        # Vertices of the tetrahedron
        vertices = [
            (0, 0, height * 2/3),  # Top vertex (apex)
            (base_radius, 0, -height / 3),  # Base vertices
            (-base_radius / 2, base_radius * np.sqrt(3) / 2, -height / 3),
            (-base_radius / 2, -base_radius * np.sqrt(3) / 2, -height / 3)
        ]

        return vertices

    @staticmethod
    def generate_sphere(radius, latitude_divisions, longitude_divisions):
        coordinates = []

        for lat in range(latitude_divisions):
            for lon in range(longitude_divisions):
                theta = 2 * math.pi * lon / longitude_divisions
                phi = math.pi * (lat + 0.5) / latitude_divisions

                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.sin(phi) * math.sin(theta)
                z = radius * math.cos(phi)

                coordinates.append((x, y, z))

        return coordinates

    def generate_points_3d(self, center, num_points):
        """
        Function to generate random points around a center
        :param center: the coordinates of the center
        :param num_points: the number of points to generate
        :return: list of generated points
        """
        points = []
        iso_points = []
        if self.stats:
            fitted_distributions = self.fitted_distributions
            shape_x, loc_x, scale_x = fitted_distributions['cluster_std_distribution']['x']['shape'], fitted_distributions['cluster_std_distribution']['x']['loc'], fitted_distributions['cluster_std_distribution']['x']['scale']
            shape_y, loc_y, scale_y = fitted_distributions['cluster_std_distribution']['y']['shape'], fitted_distributions['cluster_std_distribution']['y']['loc'], fitted_distributions['cluster_std_distribution']['y']['scale']
            shape_z, loc_z, scale_z = fitted_distributions['cluster_std_distribution']['z']['shape'], fitted_distributions['cluster_std_distribution']['z']['loc'], fitted_distributions['cluster_std_distribution']['z']['scale']
            # Sample from the fitted distributions
            std_x = lognorm.rvs(shape_x, loc=loc_x, scale=scale_x)
            std_y = lognorm.rvs(shape_y, loc=loc_y, scale=scale_y)
            std_z = lognorm.rvs(shape_z, loc=loc_z, scale=scale_z)
            shape_comb, loc_comb, scale_comb = fitted_distributions['cluster_std_combined']['shape'], fitted_distributions['cluster_std_combined']['loc'], fitted_distributions['cluster_std_combined']['scale']
            std_comb = lognorm.rvs(shape_comb, loc_comb, scale_comb)
        else:
            photon_count = np.random.randint(1000, 5000)
            std_x = 10 / np.sqrt(photon_count) * 1.5
            std_y = 10 / np.sqrt(photon_count) * 1.5
        for i in range(num_points):
                #r = random.uniform(0, radius)
                #r = np.clip(np.random.exponential(self.k * self.side), 0, self.side)
                #if self.stats:
                #    shape_r = fitted_distributions['r_distribution']['shape']
                #    loc_r = fitted_distributions['r_distribution']['loc']
                #    scale_r = fitted_distributions['r_distribution']['scale']

                #    shape_r *= 0.8  # shrink shape by 20%
                #    scale_r *= 0.8  # shrink scale by 20%
                #    r = lognorm.rvs(shape_r, loc=loc_r, scale=scale_r)

                # else:
                #    r = np.clip(np.random.exponential(self.side), 0, self.side)
                #theta = np.random.uniform(0, 2 * math.pi)
                #phi = np.random.uniform(0, math.pi)

                photon_count = np.random.randint(1000, 5000)
                std_xy = 10 / np.sqrt(photon_count) * 1.5  # in nm
                # precision_z = precision_xy * 2.5
                #combined_std = np.sqrt(std_dev**2 + std_xy**2)
                #std_z = np.mean([std_x, std_y])
                # introduce linker error (?)
                #x = np.random.normal(center[0] + r * math.sin(phi) * math.cos(theta), std_x)
                #y = np.random.normal(center[1] + r * math.sin(phi) * math.sin(theta), std_y)
                #z = np.random.normal(center[2] + r * math.cos(phi), std_z)
                # simulate anisotropic point cloud
                x = np.random.normal(center[0], std_x)
                y = np.random.normal(center[1], std_y)
                z = np.random.normal(center[2], std_z)
                # simulate isotropic point cloud
                x_iso = np.random.normal(center[0], std_comb)
                y_iso = np.random.normal(center[1], std_comb)
                z_iso = np.random.normal(center[2], std_comb)

                # ensure sample std
                #sim_std = np.std(np.array([[x, y, z]]), axis=0)
                #scale_factor = mean_sample_std / sim_std

                # Apply scaling
                #x = center[0] + (x - center[0]) * scale_factor[0]
                #y = center[1] + (y - center[1]) * scale_factor[1]
                #z = center[2] + (z - center[2]) * scale_factor[2]

                #points.append((x, y, z, overall_uncertainty_xy, uncertainty_z))

                # not implemented rigurously
                uncertainty_factor_xy = np.random.uniform(0, 0.15)
                uncertainty_factor_z = np.random.uniform(0, 0.4)
                uncertainty_x = abs(uncertainty_factor_xy)
                uncertainty_y = abs(uncertainty_factor_xy)
                uncertainty_z = abs(uncertainty_factor_z)

                # Choose the maximum uncertainty as the point's overall uncertainty
                overall_uncertainty_xy = np.average([uncertainty_x, uncertainty_y])

                points.append((x, y, z, overall_uncertainty_xy, uncertainty_z))
                iso_points.append((x_iso, y_iso, z_iso, overall_uncertainty_xy, uncertainty_z))

        return points, iso_points

    def generate_one_dna_origami_smlm(self, sampled_cluster_sizes):
        # old version: model_structure: list, radius: int, num_localizations: int):
        """
        Function to generate a dna origami similar structure as from single-molecule localization microscopy (smlm)
        :param model_structure: the initial coordinates around which to generate random localizations. Typically list of tuples containing the x,y,z coordinates
        :param radius: the radius in which the localizations should be generated. The localizations will
        be generated inside the radius as, not only at the radius
        :param num_points: the number of points to generate
        :return: returns list of generated localizations around the initial coordinates
        """

        #random_weights = np.random.random(len(self.model_structure))  # Random values for each corner
        #probabilities = random_weights / np.sum(random_weights)
        #points_per_corner = np.random.multinomial(num_points, probabilities)
        #points_per_corner = np.random.multinomial(num_points, np.ones(len(self.model_structure))/len(self.model_structure))

        #radius = int(random.choice(range(1, self.radius)))
#        for corner in self.model_structure:
#            self.dna_origami.append(self.generate_points_3d(corner))
#            self.dna_origami_list.append(self.dna_origami)
        all_points = []
        all_iso_points = []
        for corner, num_points in zip(self.model_structure, sampled_cluster_sizes):
            points, iso_points = self.generate_points_3d(corner, int(num_points))

            # Collect them into the master lists
            all_points.extend(points)
            all_iso_points.extend(iso_points)
            #self.dna_origami.extend(self.generate_points_3d(corner, int(num_points)))
            self.dna_origami = all_points
            self.dna_origami_iso = all_iso_points
        return self.dna_origami, self.dna_origami_iso

    def generate_all_dna_origami_smlm_samples(self):
        #np.random.seed(42)
        fitted_distributions = self.fitted_distributions
        shape, loc, scale = (fitted_distributions['cluster_size_distribution']['shape'],
                             fitted_distributions['cluster_size_distribution']['loc'],
                             fitted_distributions['cluster_size_distribution']['scale'])
        for i in range(self.number_samples):
            if self.apply_rotation:
                self.rotate_model()
            #if self.stats:
            #    total_points_list = [sum(points['cluster_size'] for points in statss.values()) for statss in self.stats]
                #num_points_list = [points['num_points'] for statss in self.stats for points in statss.values()]
            #    # Calculate mean (mu)
            #    mu = np.mean(total_points_list)
            #    # Calculate standard deviation (sigma)
            #    sigma = np.std(total_points_list)
            #    num_points = int(np.random.normal(mu, sigma))
            #else:
            #    num_points = np.random.randint(400, 1400)

            # Force loc=0 to prevent shifting
            sampled_cluster_sizes = lognorm.rvs(shape, loc=loc, scale=scale, size=len(self.model_structure))
            self.generate_one_dna_origami_smlm(sampled_cluster_sizes)
            self.save_samples(i)
            self.save_model_structure(i)
            #self.dna_origami = []
        sim_stats = load_and_analyze_point_clouds(os.path.join(self.base_folder, 'aniso'))
        sim_stats_dict = extract_stats(sim_stats)
        exp_stats_dict = extract_stats(self.stats)
        ks_tests = {
            "p_x": ks_2samp(sim_stats_dict['cluster_std_x'], exp_stats_dict['cluster_std_x']),
            "p_y": ks_2samp(sim_stats_dict['cluster_std_y'], exp_stats_dict['cluster_std_y']),
            "p_z": ks_2samp(sim_stats_dict['cluster_std_z'], exp_stats_dict['cluster_std_z']),
            "num_points": ks_2samp(sim_stats_dict['num_points'], exp_stats_dict['num_points']),
            "cluster_sizes": ks_2samp(sim_stats_dict['cluster_sizes'], exp_stats_dict['cluster_sizes'])
        }
        return ks_tests

    def rotate_model(self):
        random_rotation = R.random()
        rotated_model = random_rotation.apply(np.array(self.model_structure))
        #rotated_model = [tuple(corner) for corner in rotated_model]
        self.model_structure = rotated_model
        self.current_rotation = random_rotation

    def save_model_structure(self, index: int):
        df_model_structure = pd.DataFrame(self.model_structure, columns=['x', 'y', 'z'])

        # Define model directory path
        model_dir = os.path.join(self.base_folder, 'model')
        # Check if directory exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Define file path correctly
        file_path = os.path.join(model_dir, f'model_structure_{index}.csv')

        # Save DataFrame to CSV
        df_model_structure.to_csv(file_path, index=False)

    def save_samples(self, index: int):
        # flattened_list = [element for sublist in zip(*self.dna_origami) for element in sublist]
        #df = pd.DataFrame(self.dna_origami, columns=['x', 'y', 'z', 'sigma_xy', 'sigmaz'])
        # df_smlm_sample.to_csv(self.base_folder + '/smlm_sample' + str(index) + '.csv', index=False)

        #if hasattr(self, 'current_rotation'):  # We'll add this attribute in rotate_model
        #    quat = self.current_rotation.as_quat()  # Gets 4 values [x,y,z,w]
            # Add rotation columns to every row
        #    df[['rot_x', 'rot_y', 'rot_z', 'rot_w']] = quat

        # Create a DataFrame for the anisotropic data
        df_aniso = pd.DataFrame(
            self.dna_origami,
            columns=['x', 'y', 'z', 'sigma_xy', 'sigmaz']
        )

        # Create a DataFrame for the isotropic data
        df_iso = pd.DataFrame(
            self.dna_origami_iso,
            columns=['x', 'y', 'z', 'sigma_xy', 'sigmaz']
        )

        # If there's a current rotation, add the quaternion columns to both
        if hasattr(self, 'current_rotation'):
            quat = self.current_rotation.as_quat()  # [x, y, z, w]
            df_aniso[['rot_x', 'rot_y', 'rot_z', 'rot_w']] = quat
            df_iso[['rot_x', 'rot_y', 'rot_z', 'rot_w']] = quat

        # Define the directory for saving isotropic samples
        aniso_dir = os.path.join(self.base_folder, 'aniso')
        # Create the directory if it doesn't exist
        os.makedirs(aniso_dir, exist_ok=True)
        file_path_iso = os.path.join(aniso_dir, f'smlm_sample{index}.csv')
        # Save anisotropic data
        df_aniso.to_csv(file_path_iso, index=False)

        # Define the directory for saving isotropic samples
        iso_dir = os.path.join(self.base_folder, 'iso')
        # Create the directory if it doesn't exist
        os.makedirs(iso_dir, exist_ok=True)
        # Define the file path
        file_path_iso = os.path.join(iso_dir, f'smlm_sample{index}.csv')
        # Save the isotropic data
        df_iso.to_csv(file_path_iso, index=False)

        #df.to_csv(self.base_folder + '/smlm_sample' + str(index) + '.csv', index=False)

    #TODO: move to visualization
    @staticmethod
    def plot_singular_structure(corners, color='blue'):
        """"
        :param corners: list of tuple containing the coordinates
        :param color: string describing the desired color to use for plotting, default is blue
        """
        matplotlib.use('TkAgg')
        if len(plt.get_fignums()) == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = plt.gca()
        if 'tuple' in type(corners):
            x = [corner[0] for corner in corners]
            y = [corner[1] for corner in corners]
            z = [corner[2] for corner in corners]
        elif 'DataFrame' in type(corners):
            x = corners['x']
            y = corners['y']
            z = corners['z']
        else:
            raise FileNotFoundError('Data is empty.')
        ax.scatter(x, y, z, color=color)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def plot_dna_origami_smlm(self, model_structure, dna_origami):
        """
        Plots the dna origami final structure
        """
        self.plot_singular_structure(model_structure, 'red')
        for origami in dna_origami:
            self.plot_singular_structure(origami)

def _remove_corners(point_cloud):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, alpha=0.7)
    cluster_labels = clusterer.fit_predict(point_cloud)

    valid_clusters = [c for c in set(cluster_labels) if c != -1]
    if np.random.rand() < 0.8 and valid_clusters:
        cluster_to_remove = np.random.choice(valid_clusters)
        return point_cloud[cluster_labels != cluster_to_remove]
    return point_cloud

def lognorm_fit_with_median_correction(data, floc=0):
    """
    Fits a lognormal distribution to 'data' with an optional loc=floc,
    then uses the median-based scale correction.
    Returns shape, loc, scale_corrected.
    """
    shape, loc, scale = lognorm.fit(data, floc=floc)
    median_ = np.median(data)
    # Correct scale so that lognorm mean & median line up
    # (this addresses small offsets the built-in MLE can produce)
    scale_corrected = median_ / np.exp(shape**2 / 2)
    return shape, loc, scale_corrected

def lognormal_params(mean_X, std_X):
    """
    Convert mean and standard deviation of a lognormal distribution
    to the corresponding normal distribution parameters.

    Parameters:
    mean_X (float): Mean of the lognormal distribution
    std_X (float): Standard deviation of the lognormal distribution

    Returns:
    tuple: (mu, sigma) parameters for the underlying normal distribution
    """
    mu = np.log(mean_X ** 2 / np.sqrt(mean_X ** 2 + std_X ** 2))
    sigma = np.sqrt(np.log(1 + (std_X ** 2 / mean_X ** 2)))
    return mu, sigma

def truncated_lognorm_rvs(shape, loc, scale, lower, upper, size=1):
    """
    Sample from a lognormal distribution truncated to [lower, upper].
    Uses simple one-by-one rejection sampling.
    Returns a 1D array of length `size`.
    """
    from scipy.stats import lognorm

    samples = []
    while len(samples) < size:
        # Sample just one value
        candidate = lognorm.rvs(shape, loc=loc, scale=scale, size=1)[0]
        # Check if it's within [lower, upper]
        if lower <= candidate <= upper:
            samples.append(candidate)

    return np.array(samples)