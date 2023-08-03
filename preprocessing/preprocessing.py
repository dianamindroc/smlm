from typing import Tuple
import pandas as pd
import open3d as o3d
import numpy as np
import hdbscan
import os
import shutil
import glob
import sklearn.neighbors as sngh
import networkx as nx
import torch
from torch_geometric.data import Data


class Preprocessing:
    """
    Class for preprocessing SMLM data.
    """

    def __init__(self, path: str):
        """
        Initialization function
        :param path: data in the form of path to the folder containing both train and test data in train and test folders.
        """
        self.graph = None
        self.path = path
        #self.downsampled_data = pd.DataFrame()
        #self.denoised_data = pd.DataFrame()
        self.outliers = np.zeros(10)
        self.graph_array = np.zeros(10)
        if os.path.exists(os.path.join(os.path.join(self.path, 'preprocessed'))):
            print('The data might have been preprocessed already. The folder exists.')
        else:
            shutil.copytree(os.path.join(self.path, 'train'), os.path.join(os.path.join(self.path, 'preprocessed'), 'train'))
            shutil.copytree(os.path.join(self.path, 'test'), os.path.join(os.path.join(self.path, 'preprocessed'), 'test'))
        self.train_path = os.path.join(os.path.join(self.path, 'preprocessed'), 'train')
        self.test_path = os.path.join(os.path.join(self.path, 'preprocessed'), 'test')
        self.downsampled = False
        self.denoised = False

    def downsample(self, factor: int, name: str = 'Localizations', suffix: str ='.txt'):
        """
        Method for downsampling the data uniformly to retain geometry. Overwrites the data in the preprocessed data folder.
        :param factor: the factor by which to downsample the pointcloud
        :return: None
        """
        for file in glob.glob(self.train_path+'/*/*' + name + suffix):
            pc = o3d.io.read_point_cloud(file, format='xyz')
            down_pc = pc.uniform_down_sample(factor)
            downsampled_data = pd.DataFrame(np.asarray(down_pc.points), columns=['x', 'y', 'z'])
            downsampled_data.to_csv(file.split('.')[0] + '.csv', index=False)
        for file in glob.glob(self.test_path+'/*/*' + name + suffix):
            pc = o3d.io.read_point_cloud(file, format='xyz')
            down_pc = pc.uniform_down_sample(factor)
            downsampled_data = pd.DataFrame(np.asarray(down_pc.points), columns=['x', 'y', 'z'])
            downsampled_data.to_csv(file.split('.')[0] + '.csv', index=False)
        self.downsampled = True

    def denoise(self, min_cluster_size=10, min_samples=5, outlier_threshold=0.8, suffix: str = '.txt', name: str = 'Localizations') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method for denoising the data using hdbscan. Overwrites the data in the preprocessed data folder.
        :param min_cluster_size: the desired minimum size of the cluster. Can be application-dependent
        :param min_samples: the desired minimum of samples in a cluster
        :param outlier_threshold: the threshold for the quantile for separating outliers from data
        :return: denoised data in the form of pandas DataFrame, outliers
        """
        for file in glob.glob(self.train_path+'/*/*' + name + suffix):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True,
                                        algorithm='best', alpha=0.7, metric='euclidean')
            data = pd.read_csv(file)
            clusterer.fit(data)
            threshold = pd.Series(clusterer.outlier_scores_).quantile(outlier_threshold)
            outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
            denoised_data = data.drop(data.index[outliers])
            #downsampled_data = pd.DataFrame(np.asarray(down_pc.points), columns=['x', 'y', 'z'])
            denoised_data.to_csv(file.split('.')[0] + '.csv', index=False)
            self.denoised_data = denoised_data
        return denoised_data, outliers

    def get_graphs(self, n_neighbors: int, algorithm: str ='kd_tree', name: str = 'Localizations', suffix: str ='.csv') -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to get data in form of graphs
        :param n_neighbors: number of neighbors to be considered when building the graph
        :param algorithm: algorithm to be used for graph building, default is kd_tree
        :return: graph as array, graph as kneighbors graph
        """
        for file in glob.glob(self.train_path+'/*/*' + name + suffix):
            transformer = sngh.KNeighborsTransformer(n_neighbors=n_neighbors, algorithm=algorithm)
            transformer.fit_transform(pd.read_csv(file))
            graph = transformer.kneighbors_graph()
            nx_graph = nx.from_numpy_array(graph.toarray())
            edge_index = np.transpose(nx_graph.edges)
            data = Data(x=graph,
                    edge_index=edge_index,
                    )
            torch.save(data, file.split('.')[0] + '.pt')
        for file in glob.glob(self.test_path+'/*/*' + name + suffix):
            transformer = sngh.KNeighborsTransformer(n_neighbors=n_neighbors, algorithm=algorithm)
            transformer.fit_transform(pd.read_csv(file))
            graph = transformer.kneighbors_graph()
            nx_graph = nx.from_numpy_array(graph.toarray())
            edge_index = np.transpose(nx_graph.edges)
            data = Data(x=graph,
                    edge_index=edge_index,
                    )
            torch.save(data, file.split('.')[0] + '.pt')
        #return self.graph, self.edge_index
#
#     def _get_graph_(path, name, suffix):
#         """
#         Inside method for getting the graphs
#         """
#         for file in glob.glob(path+'/*/*' + name + suffix):
#             transformer = sngh.KNeighborsTransformer(n_neighbors=n_neighbors, algorithm=algorithm)
#             transformer.fit_transform(pd.read_csv(file))
#             graph = transformer.kneighbors_graph()
#             nx_graph = nx.from_numpy_array(graph.toarray())
#             edge_index = np.transpose(nx_graph.edges)
#             data = Data(x=graph,
#                     edge_index=edge_index,
#                     )
#             torch.save(data, file.split('.')[0] + '.pt')

def downsample_one(file_path, downsample_factor):
    pc = o3d.io.read_point_cloud(file_path, format='xyz')
    down_pc = pc.uniform_down_sample(factor)
    downsampled_data = pd.DataFrame(np.asarray(down_pc.points), columns=['x', 'y', 'z'])
    downsampled_data.to_csv(file_path.split('.')[0] + '.csv', index=False)

def denoise_one(file, min_cluster_size, min_samples, outlier_threshold):
    if isinstance(file, pd.DataFrame):
        data = file
    elif isinstance(file, str):
        data = pd.read_csv(file)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True,
                                        algorithm='best', alpha=0.7, metric='euclidean')
    clusterer.fit(data)
    threshold = pd.Series(clusterer.outlier_scores_).quantile(outlier_threshold)
    outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
    denoised_data = data.drop(data.index[outliers])
    return denoised_data



