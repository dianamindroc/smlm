import matplotlib.pyplot as plt
import matplotlib
import random
import math


def generate_cube(x: int, y: int, z: int, size: int):
    """"
    Generates a cube with the first corner at the x,y,z coordinates and a specific size
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param size: the size of one edge
    :return: returns a list of tuples containing the coordinates
    """
    corners = [(x, y, z),
               (x+size, y, z),
               (x+size, y+size, z),
               (x, y+size, z),
               (x, y, z+size),
               (x+size, y, z+size),
               (x+size, y+size, z+size),
               (x, y+size, z+size)]
    return corners


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
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    z = [corner[2] for corner in corners]
    ax.scatter(x, y, z, color=color)


def generate_points_3d(center, radius, num_points):
    """"
    Function to generate random points around a center
    :param center: the coordinates of the center
    :param radius: the radius in which the points should be generated. The points will be generated inside the radius as well, not only at the radius
    :param num_points: the number of points to generate
    :return: list of generated points
    """
    points = []
    for i in range(num_points):
        r = random.uniform(0, radius)
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        x = center[0] + r * math.sin(phi) * math.cos(theta)
        y = center[1] + r * math.sin(phi) * math.sin(theta)
        z = center[2] + r * math.cos(phi)
        points.append((x, y, z))
    return points


def generate_dna_origami_smlm(model_structure: list, radius: int, num_localizations: int):
    """"
    Function to generate a dna origami similar structure as from single-molecule localization microscopy (smlm)
    :param model_structure: the initial coordinates around which to generate random localizations. Typically list of tuples containing the x,y,z coordinates
    :param radius: the radius in which the localizations should be generated. The localizations will
    be generated inside the radius as, not only at the radius
    :param num_localizations: the desired number of localizations to generate
    :return: returns list of generated localizations around the initial coordinates
    """
    dna_origami = []
    for corner in model_structure:
        dna_origami.append(generate_points_3d(corner, radius, num_localizations))
    return dna_origami


def plot_dna_origami_smlm(model_structure, dna_origami):
    """
    Plots the dna origami final structure
    """
    plot_singular_structure(model_structure, 'red')
    for origami in dna_origami:
        plot_singular_structure(origami)
