import matplotlib.pyplot as plt
import matplotlib
import random
import math


def generate_cube(x, y, z, size):
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
    # Generate random 3D points around center coordinates within radius of 10
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


def generate_dna_origami_smlm(model_structure, radius, num_points):
    dna_origami = []
    for corner in model_structure:
        dna_origami.append(generate_points_3d(corner, radius, num_points))
    return dna_origami


def plot_dna_origami_smlm(model_structure, dna_origami):
    plot_singular_structure(model_structure, 'red')
    for origami in dna_origami:
        plot_singular_structure(origami)
