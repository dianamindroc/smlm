import matplotlib.pyplot as plt
import matplotlib
import random
import math
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R


class SMLMDnaOrigami:
    """
    Class to simulate a DNA origami
    :param struct_type: string describing the type of structures. Choose from box, pyramid, tetrahedron, sphere.
    :param number_dna_origami_samples: Number of DNA origami samples to simulate
    :param stats: Dictionary of statistics about the DNA origami to use during simulation
    :param apply_rotation: Boolean to apply the rotation to the DNA origami model structure
    """

    def __init__(self, model:str, number_dna_origami_samples: int, stats=None, apply_rotation=True):
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
        df = pd.read_csv(model)
        self.model_structure = [tuple(row) for row in df[['x', 'y', 'z']].values]
        #self.radius = int(input("What is the maximum radius in which to generate SMLM samples?"))
        #self.number_localizations, self.percentage = [int(x) for x in input("How many localizations to generate around each center and with which variation (separate by space)?").split()]
        #self.uncertainty_factor = int(input("What is the uncertainty factor for the localizations?"))
        self.base_folder = input("Where to save the generated samples?")
        print('Going to generate ' + str(number_dna_origami_samples) + ' samples for the given model.')
        self.generate_all_dna_origami_smlm_samples()


    def generate_points_3d(self, center, num_points):
        """
        Function to generate random points around a center
        :param center: the coordinates of the center
        :param num_points: the number of points to generate
        :return: list of generated points
        """
        points = []

        if self.stats:
            std_dev_df = pd.DataFrame([stats['std_dev'] for stats in self.stats], columns=['x', 'y', 'z'])
            overall_std_dev = std_dev_df.mean()
            std_x, std_y, std_z = overall_std_dev['x'], overall_std_dev['y'], overall_std_dev['z']
        for i in range(num_points):
            #r = random.uniform(0, radius)
            r = np.random.exponential(1.0)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)

            #gaussian_values = np.random.normal(loc=0, scale=1, size=num_points)

            #r *= gaussian_values

            #x = center[0] + r * math.sin(phi) * math.cos(theta)
            #y = center[1] + r * math.sin(phi) * math.sin(theta)
            #z = center[2] + r * math.cos(phi)

            if self.stats:
                std_x = std_x
                std_y = std_y
                std_z = std_z
            else:
                photon_count = np.random.randint(1000, 5000)
                std_x = 10 / np.sqrt(photon_count) * 1.5
                std_y = 10 / np.sqrt(photon_count) * 1.5
                std_z = 10 / np.sqrt(photon_count) * 1.5# in nm
                # precision_z = precision_xy * 2.5

            x = np.random.normal(center[0] + r * math.sin(phi) * math.cos(theta), std_x)
            y = np.random.normal(center[1] + r * math.sin(phi) * math.sin(theta), std_y)
            z = np.random.normal(center[2] + r * math.cos(phi), std_z)
            uncertainty_factor_xy = np.random.uniform(0, 0.15)
            uncertainty_factor_z = np.random.uniform(0, 0.4)
            uncertainty_x = abs(r * math.sin(phi) * math.cos(theta) * uncertainty_factor_xy)
            uncertainty_y = abs(r * math.sin(phi) * math.sin(theta) * uncertainty_factor_xy)
            uncertainty_z = abs(r * math.cos(phi) * uncertainty_factor_z)

            # Choose the maximum uncertainty as the point's overall uncertainty
            overall_uncertainty_xy = np.average([uncertainty_x, uncertainty_y])

            points.append((x, y, z, overall_uncertainty_xy, uncertainty_z))
        return points

    def generate_one_dna_origami_smlm(self, num_points):
        # old version: model_structure: list, radius: int, num_localizations: int):
        """
        Function to generate a dna origami similar structure as from single-molecule localization microscopy (smlm)
        :param model_structure: the initial coordinates around which to generate random localizations. Typically list of tuples containing the x,y,z coordinates
        :param radius: the radius in which the localizations should be generated. The localizations will
        be generated inside the radius as, not only at the radius
        :param num_points: the number of points to generate
        :return: returns list of generated localizations around the initial coordinates
        """

        points_per_corner = np.random.multinomial(num_points, np.ones(len(self.model_structure))/len(self.model_structure))

        #radius = int(random.choice(range(1, self.radius)))
#        for corner in self.model_structure:
#            self.dna_origami.append(self.generate_points_3d(corner))
#            self.dna_origami_list.append(self.dna_origami)
        for corner, num_points in zip(self.model_structure, points_per_corner):
            self.dna_origami.extend(self.generate_points_3d(corner, num_points))
        return self.dna_origami

    def generate_all_dna_origami_smlm_samples(self):
        for i in range(self.number_samples):
            if self.apply_rotation:
                self.rotate_model()
            if self.stats:
                num_points_list = [stats['num_points'] for stats in self.stats]
                # Calculate mean (mu)
                mu = np.mean(num_points_list)
                # Calculate standard deviation (sigma)
                sigma = np.std(num_points_list)
                num_points = int(np.random.normal(mu, sigma))
            else:
                num_points = np.random.randint(400, 1400)
            self.generate_one_dna_origami_smlm(num_points)
            self.save_samples(i)
            self.save_model_structure(i)
            self.dna_origami = []

    def rotate_model(self):
        random_rotation = R.random()
        rotated_model = random_rotation.apply(np.array(self.model_structure))
        rotated_model = [tuple(corner) for corner in rotated_model]
        self.model_structure = rotated_model

    def save_model_structure(self, index: int):
        df_model_structure = pd.DataFrame(self.model_structure, columns=['x', 'y', 'z'])
        df_model_structure.to_csv(self.base_folder + '/model_structure' + str(index) + '.csv', index=False)

    def save_samples(self, index: int):
        # flattened_list = [element for sublist in zip(*self.dna_origami) for element in sublist]
        df_smlm_sample = pd.DataFrame(self.dna_origami, columns=['x', 'y', 'z', 'sigma_xy', 'sigmaz'])
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