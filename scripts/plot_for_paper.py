import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch

cloud1 = o3d.io.read_point_cloud("/home/dim26fa/coding/testing/logs_pcn_20240420_134549/058_gt.ply")
cloud2 = o3d.io.read_point_cloud("/home/dim26fa/coding/testing/logs_pcn_20240420_134549/058_input.ply")
cloud3 = o3d.io.read_point_cloud("/home/dim26fa/coding/testing/logs_pcn_20240420_134549/058_output.ply")
#058, 104, 130, 025
points1 = np.asarray(cloud1.points)
points2 = np.asarray(cloud2.points)
points3 = np.asarray(cloud3.points)


## tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tens1 = torch.tensor(points1).to(device).float().unsqueeze(0)
tens2 = torch.tensor(points2).to(device).float().unsqueeze(0)

points2 = superParticle[5]

fig = plt.figure(dpi=300)

ax = fig.add_subplot(131, projection='3d')
ax.scatter(points1[:,0], points1[:,1], points1[:,2], depthshade=False, color='lightseagreen', s=1)
ax.title.set_text('Ground Truth')
ax.set_axis_off()

ax = fig.add_subplot(132, projection='3d')
ax.scatter(points2[:,0], points2[:,1], points2[:,2], depthshade=False, color='lightseagreen', s=1)
ax.title.set_text('Input')
ax.set_axis_off()

ax = fig.add_subplot(133, projection='3d')
ax.scatter(points3[:,0], points3[:,1], points3[:,2], depthshade=False, color='lightseagreen', s=1)
ax.title.set_text('Output')
ax.set_axis_off()

plt.savefig('/home/dim26fa/data/to_plot/figure7_smallpoints.png')
plt.show()

import glob
from model_architectures.losses import cd_loss_l1
file_list = glob.glob('/home/dim26fa/coding/testing/logs_pcn_20240420_134549/*.ply')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
losses = []
for file in file_list:
    if 'gt' in file:
        gt = o3d.io.read_point_cloud(file)
        points1 = np.asarray(gt.points)
        tens1 = torch.tensor(points1).to(device).float().unsqueeze(0)
    if 'output' in file:
        output = o3d.io.read_point_cloud(file)
        points2 = np.asarray(output.points)
        tens2 = torch.tensor(points2).to(device).float().unsqueeze(0)
    loss = cd_loss_l1(tens1, tens2)
    losses.append(loss.item())

cloud1 = o3d.io.read_point_cloud("/home/dim26fa/coding/testing/logs_pcn_20240420_134549/104_gt.ply")
cloud2 = o3d.io.read_point_cloud("/home/dim26fa/coding/testing/logs_pcn_20240420_134549/104_input.ply")
cloud3 = o3d.io.read_point_cloud("/home/dim26fa/coding/testing/logs_pcn_20240420_134549/104_output.ply")
#058, 104
points1 = np.asarray(cloud1.points)
points2 = np.asarray(cloud2.points)
points3 = np.asarray(cloud3.points)

points1 = np.array(centers_of_mass_gt[['x', 'y', 'z']])
points2 = np.array(centers_of_mass_pred[['x', 'y', 'z']])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(centers_of_mass_pred['x'], centers_of_mass_pred['y'], centers_of_mass_pred['z'], color='lightseagreen', marker='o', label='pred', s=15)
ax.scatter(centers_of_mass_gt['x'], centers_of_mass_gt['y'], centers_of_mass_gt['z'], color='black', marker='o', label='gt', s=15)
#ax.set_axis_off()
ax.grid(False)
plt.legend(loc=2, prop={'size': 12})
plt.savefig('/home/dim26fa/data/to_plot/centers2.png')
plt.show()

## tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tens1 = torch.tensor(points1).to(device).float().unsqueeze(0)
tens2 = torch.tensor(points2).to(device).float().unsqueeze(0)

squared_dist = np.sum((points1[2,:]-points2[3,:])**2, axis=0)
dist = np.sqrt(squared_dist)
dist
