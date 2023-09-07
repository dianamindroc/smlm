import matplotlib.pyplot as plt
import matplotlib as mlp
import open3d as o3d
import plotly.express as px
import plotly
#
# from helpers.readers import Reader
#
# reader = Reader('/Users/dianamindroc/Desktop/PhD/Data/suresim_simulations/IMOD_models/model_0')
# reader.get_folders('1')
# reader.set_folder(1)
# reader.get_files_from_folder()
# reader.filter('.txt')
# reader.set_file(1)
#
# data = reader.read_txt(columns=4)
# xyz = reader.extract_xyz(column_names=[0,1,2])
#
# px.scatter_3d(xyz)
#
# mlp.use('MacOSX')
#
# plt.plot(1,2)

#TODO
#clean notebook to show pipeline
#prepare simulation model example and samples - screenshot of multiple 3D plots?
#mention that I met with Matt Lycas
#show HBDBSCAN result
#show graphs example
#mention about DGCNN and PointNet - and that starting now to focus on that in details
#mention about abstract
#mention about container
#mention that i will focus now on a clustering application
#LOGML acceptance


from preprocessing.preprocessing import Preprocessing

prepp = Preprocessing('/Users/dianamindroc/Desktop/PhD/Data/suresim_simulations/IMOD_models/model_0')

prepp.downsample(100)
prepp.denoise()


import torch
from chamfer_distance import ChamferDistance
import time

chamfer_dist = ChamferDistance()

p1 = torch.rand([10,25,3])
p2 = torch.rand([10,15,3])

s = time.time()
dist1, dist2, idx1, idx2 = chamfer_dist(p1,p2)
loss = (torch.mean(dist1)) + (torch.mean(dist2))

torch.cuda.synchronize()
print(f"Time: {time.time() - s} seconds")
print(f"Loss: {loss}")

import glob
config = glob.glob('/home/squirrel/coding/smlm/configs/config.yaml')