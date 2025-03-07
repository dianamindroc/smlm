import matplotlib.pyplot as plt
import matplotlib
import random
import math
import pandas as pd
import numpy as np
from mpmath import randmatrix
from scipy.spatial.transform import Rotation as R


class SMLMDnaOrigami:
    """
    Class to simulate a DNA origami
    :param struct_type: string describing the type of structures. Choose from box, pyramid, tetrahedron, sphere.
    :param number_dna_origami_samples: Number of DNA origami samples to simulate
    :param stats: Dictionary of statistics about the DNA origami to use during simulation
    :param apply_rotation: Boolean to apply the rotation to the DNA origami model structure
    """

    def __init__(self, struct_type: str, number_dna_origami_samples: int, stats=None, apply_rotation=True):
        self.current_rotation = None
        self.struct_type = struct_type
        self.number_samples = number_dna_origami_samples
        self.dna_origami_list = []
        self.dna_origami = []
        self.apply_rotation = apply_rotation
        if stats is not None:
            self.stats = stats
            print('You provided a stats file. Simulating with prior knowledge.')
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
            self.side = int(input("Enter tetrahedron side length: "))
            self.k = float(input("Enter scaling factor for std_dev (relative to side length): "))
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
        self.base_folder = input("Where to save the generated samples?")
        print('Going to generate ' + str(number_dna_origami_samples) + ' samples for ' + struct_type + '.')
        self.generate_all_dna_origami_smlm_samples()

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

        if self.stats:
            cluster_std_df = pd.DataFrame([points['cluster_std'] for statss in self.stats for points in statss.values()], columns=['x', 'y', 'z'])
            overall_std_dev = cluster_std_df.mean()
            std_x, std_y, std_z = overall_std_dev['x'], overall_std_dev['y'], overall_std_dev['z']
            sample_std_df = pd.DataFrame(
                [points['sample_std'] for statss in self.stats for points in statss.values()], columns=['x', 'y', 'z'])
            mean_sample_std = sample_std_df.mean()
        else:
            photon_count = np.random.randint(1000, 5000)
            std_x = 10 / np.sqrt(photon_count) * 1.5
            std_y = 10 / np.sqrt(photon_count) * 1.5
        for i in range(num_points):
                #r = random.uniform(0, radius)
                r = np.clip(np.random.exponential(self.k * self.side), 0, self.side)
                theta = random.uniform(0, 2 * math.pi)
                phi = random.uniform(0, math.pi)

                photon_count = np.random.randint(1000, 5000)
                std_xy = 10 / np.sqrt(photon_count) * 1.5  # in nm
                # precision_z = precision_xy * 2.5
                #combined_std = np.sqrt(std_dev**2 + std_xy**2)
                std_z = np.mean([std_x, std_y])
                x = np.random.normal(center[0] + r * math.sin(phi) * math.cos(theta), std_x)
                y = np.random.normal(center[1] + r * math.sin(phi) * math.sin(theta), std_y)
                z = np.random.normal(center[2] + r * math.cos(phi), std_z)

                # ensure sample std
                #sim_std = np.std(np.array([[x, y, z]]), axis=0)
                #scale_factor = mean_sample_std / sim_std

                # Apply scaling
                #x = center[0] + (x - center[0]) * scale_factor[0]
                #y = center[1] + (y - center[1]) * scale_factor[1]
                #z = center[2] + (z - center[2]) * scale_factor[2]

                #points.append((x, y, z, overall_uncertainty_xy, uncertainty_z))

                uncertainty_factor_xy = np.random.uniform(0, 0.15)
                uncertainty_factor_z = np.random.uniform(0, 0.4)
                uncertainty_x = abs(r * math.sin(phi) * math.cos(theta) * uncertainty_factor_xy)
                uncertainty_y = abs(r * math.sin(phi) * math.sin(theta) * uncertainty_factor_xy)
                uncertainty_z = abs(r * math.cos(phi) * uncertainty_factor_z)

                # Choose the maximum uncertainty as the point's overall uncertainty
                overall_uncertainty_xy = np.average([uncertainty_x, uncertainty_y])

                points.append((x, y, z, overall_uncertainty_xy, uncertainty_z))


        return points

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
        for corner, num_points in zip(self.model_structure, sampled_cluster_sizes):
            self.dna_origami.extend(self.generate_points_3d(corner, int(num_points)))
        return self.dna_origami

    def generate_all_dna_origami_smlm_samples(self):
        np.random.seed(42)
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
            cluster_sizes = [pointsss['cluster_size']
                             for statss in self.stats
                             for pointsss in statss.values()
                             if 'cluster_size' in pointsss]
            cluster_sizesdf = np.array(cluster_sizes)
            mu, sigma = np.mean(cluster_sizesdf), np.std(cluster_sizesdf)
            sampled_cluster_sizes = np.random.normal(mu, sigma, len(self.model_structure))
            self.generate_one_dna_origami_smlm(sampled_cluster_sizes)
            self.save_samples(i)
            self.save_model_structure(i)
            self.dna_origami = []

    def rotate_model(self):
        random_rotation = R.random()
        rotated_model = random_rotation.apply(np.array(self.model_structure))
        rotated_model = [tuple(corner) for corner in rotated_model]
        self.model_structure = rotated_model
        self.current_rotation = random_rotation

    def save_model_structure(self, index: int):
        df_model_structure = pd.DataFrame(self.model_structure, columns=['x', 'y', 'z'])
        df_model_structure.to_csv(self.base_folder + '/model' + '/model_structure' + str(index) + '.csv', index=False)

    def save_samples(self, index: int):
        # flattened_list = [element for sublist in zip(*self.dna_origami) for element in sublist]
        df_smlm_sample = pd.DataFrame(self.dna_origami, columns=['x', 'y', 'z', 'sigma_xy', 'sigmaz'])
        # df_smlm_sample.to_csv(self.base_folder + '/smlm_sample' + str(index) + '.csv', index=False)

        if hasattr(self, 'current_rotation'):  # We'll add this attribute in rotate_model
            quat = self.current_rotation.as_quat()  # Gets 4 values [x,y,z,w]
            # Add rotation columns to every row
            df_smlm_sample['rot_x'] = quat[0]
            df_smlm_sample['rot_y'] = quat[1]
            df_smlm_sample['rot_z'] = quat[2]
            df_smlm_sample['rot_w'] = quat[3]

        df_smlm_sample.to_csv(self.base_folder + '/smlm_sample' + str(index) + '.csv', index=False)

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