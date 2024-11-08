import numpy as np
import open3d as o3d
import pandas as pd
import glob
import hdbscan
import matplotlib.pyplot as plt

from helpers.readers import read_pc

paths = glob.glob('/home/dim26fa/data/NPC_Sauer_Danush/2D/*.txt')
file = paths[2]
pc = pd.read_csv(file, delim_whitespace=True, skiprows=1, usecols=[0,1], names=['x','y'])

clusterer = hdbscan.HDBSCAN(min_cluster_size=350, min_samples=345)
labels = clusterer.fit_predict(pc)

# Filter out noise (label == -1 is considered noise in HDBSCAN)
clustered_data = pc[labels != -1].copy()
clustered_data['cluster'] = labels[labels != -1]

cluster_sizes = clustered_data['cluster'].value_counts().nlargest(10)
largest_clusters_data = clustered_data[clustered_data['cluster'].isin(cluster_sizes.index)]


###

path = '/home/dim26fa/data/NPC_sim_Theiss/mag1rsigma0/sim_python/smlm_sample50.csv'
data = pd.read_csv(path)

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=95)
labels = clusterer.fit_predict(data[['xnm', 'ynm','znm']])

clustered_data = data[labels != -1].copy()
clustered_data['cluster'] = labels[labels != -1]

####
paths = glob.glob('/home/dim26fa/data/NPC_Sauer_Danush/2D/*.txt')
file = paths[14]
pc = pd.read_csv(file, delim_whitespace=True, skiprows=1, usecols=[0,1], names=['x','y'])

clusterer = hdbscan.HDBSCAN(min_cluster_size=350, min_samples=345)
labels = clusterer.fit_predict(pc)
clustered_data = pc[labels != -1].copy()
clustered_data['cluster'] = labels[labels != -1]
centroids = clustered_data.groupby('cluster')[['x', 'y']].mean()

# Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

# Scatter plot with cluster labels
scatter = ax.scatter(clustered_data['x'], clustered_data['y'],
                     c=clustered_data['cluster'], s=0.5, cmap='tab20', alpha=0.7)
ax.set_title('3D SMLM Data with Cluster Labels')
ax.set_xlabel('X Coordinate (nm)')
ax.set_ylabel('Y Coordinate (nm)')

# Label each cluster at its centroid
for cluster_id, (x, y) in centroids.iterrows():
    ax.text(x, y, str(cluster_id), fontsize=3, color="black",
            ha='center', va='center')

# Save and display the plot
plt.savefig('/home/dim26fa/data/NPC_Sauer_Danush/2D/clustering/'+ file.split('/')[-1].split('.')[0] + '.png', dpi=600)
plt.show()

from npc_smlm_averaging import save_and_plot_clusters_3d
import os

save_and_plot_clusters_3d(clustered_data, os.path.dirname(path) + '/clusters')
