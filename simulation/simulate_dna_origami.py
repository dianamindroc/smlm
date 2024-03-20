import matplotlib.pyplot as plt
import matplotlib
import random
import math
import pandas as pd
import numpy as np

class SMLMDnaOrigami:
    def __init__(self, struct_type: str, number_dna_origami_samples: int, save_model=True, anisotropic_axis='z', anisotropy_factor=3.0):
        self.struct_type = struct_type
        self.number_samples = number_dna_origami_samples
        self.dna_origami_list = []
        self.dna_origami = []
        if struct_type == 'cube':
            x, y, z, size = [int(x) for x in input("Enter cube coordinates and maximum size (separate by space): ").split()]
            self.model_structure = self.generate_cube(x, y, z, size)
        elif struct_type == 'pyramid':
            base, height = [int(x) for x in input("Enter pyramid base edge size and height (separate by space): ").split()]
            self.model_structure = self.generate_pyramid(base, height)
        elif struct_type == 'sphere':
            radius, latitude_divisions, longitude_divisions = [int(x) for x in input("Enter sphere radius, "
                                                                                     "nr. of latitude_divisions "
                                                                                     "and nr. of longitude_divisions: ").split()]
            self.model_structure = self.generate_sphere(radius, latitude_divisions, longitude_divisions)
        else:
            raise NotImplementedError("Only cube and pyramid supported at the moment :)")
        self.anisotropic_axis = anisotropic_axis
        self.anisotropy_factor = float(input("Enter anisotropy factor (bigger than 1.0): "))
        self.radius = int(input("What is the maximum radius in which to generate SMLM samples?"))
        self.number_localizations, self.percentage = [int(x) for x in input("How many localizations to generate around each center and with which variation (separate by space)?").split()]
        self.base_folder = input("Where to save the generated samples?")
        if save_model:
            self.save_model_structure()
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

    @staticmethod
    def generate_points_3d(center, radius, num_points, percentage, anisotropy_factor=3.0, anisotropic_axis='z'):
        """"
        Function to generate random points around a center
        :param center: the coordinates of the center
        :param radius: the radius in which the points should be generated. The points will be generated inside the radius as well, not only at the radius
        :param num_points: the number of points to generate
        :param percentage: percentage of variation desired in the data
        :return: list of generated points
        """
        points = []
        #threequarter = int(3/4*len(num_points))
        variation = percentage
        new_loc = int(num_points + random.uniform(-variation, variation))
        for i in range(new_loc):
            r = random.uniform(0, radius)
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, math.pi)

            #gaussian_values = np.random.normal(loc=0, scale=1, size=num_points)

            #r *= gaussian_values

            x = center[0] + r * math.sin(phi) * math.cos(theta)
            y = center[1] + r * math.sin(phi) * math.sin(theta)
            z = center[2] + r * math.cos(phi)

            # Apply anisotropy scaling factor to the selected axis
            if anisotropic_axis == 'x':
                x *= anisotropy_factor
            elif anisotropic_axis == 'y':
                y *= anisotropy_factor
            elif anisotropic_axis == 'z':
                z *= anisotropy_factor

            # Translate the point by the center coordinates
            x += center[0]
            y += center[1]
            z += center[2]
            points.append((x, y, z))
        return points

    def generate_one_dna_origami_smlm(self):
        # old version: model_structure: list, radius: int, num_localizations: int):
        """"
        Function to generate a dna origami similar structure as from single-molecule localization microscopy (smlm)
        :param model_structure: the initial coordinates around which to generate random localizations. Typically list of tuples containing the x,y,z coordinates
        :param radius: the radius in which the localizations should be generated. The localizations will
        be generated inside the radius as, not only at the radius
        :param num_localizations: the desired number of localizations to generate
        :return: returns list of generated localizations around the initial coordinates
        """
        radius = int(random.choice(range(1, self.radius)))
        for corner in self.model_structure:
            self.dna_origami.append(self.generate_points_3d(corner, radius, self.number_localizations, self.percentage, self.anisotropy_factor, self.anisotropic_axis))
            self.dna_origami_list.append(self.dna_origami)

    def generate_all_dna_origami_smlm_samples(self):
        for i in range(self.number_samples):
            self.generate_one_dna_origami_smlm()
            self.save_samples(i)
            self.dna_origami = []

    def save_model_structure(self):
        df_model_structure = pd.DataFrame(self.model_structure, columns=['x', 'y', 'z'])
        df_model_structure.to_csv(self.base_folder + '/model_structure.csv', index=False)

    def save_samples(self, index: int):
        flattened_list = [element for sublist in zip(*self.dna_origami) for element in sublist]
        df_smlm_sample = pd.DataFrame(flattened_list, columns=['x', 'y', 'z'])
        df_smlm_sample.to_csv(self.base_folder + '/smlm_sample' + str(index) + '.csv', index=False)

    @staticmethod
    def plot_singular_structure(corners, color='blue'):
        """"
        :param corners: list of tuple containing the coordinates
        :param color: string describing the desired color to use for plotting, default is blue
        :param axes_exists: bool explaining if there is already an axes to plot on; by default False
        :param axes: plot object to plot on, by default None
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
