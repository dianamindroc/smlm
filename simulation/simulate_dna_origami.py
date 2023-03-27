import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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

def plot_cube(corners):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]
    z = [corner[2] for corner in corners]
    ax.scatter(x, y, z)