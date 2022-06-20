from typing import Tuple
import pandas as pd
import open3d as o3d
import numpy as np
import hdbscan
import os
import shutil
import glob


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

    def downsample(self, factor: int):
        """
        Method for downsampling the data uniformly to retain geometry. Overwrites the data in the preprocessed data folder.
        :param factor: the factor by which to downsample the pointcloud
        :return: None
        """
        if not self.denoised:
            suffix = '.txt'
        else:
            suffix = '.csv'
        for file in glob.glob(self.train_path+'/*/*Localizations' + suffix):
            pc = o3d.io.read_point_cloud(file, format='xyz')
            down_pc = pc.uniform_down_sample(factor)
            downsampled_data = pd.DataFrame(np.asarray(down_pc.points), columns=['x', 'y', 'z'])
            downsampled_data.to_csv(file.split('.')[0] + '.csv', index=False)
        for file in glob.glob(self.test_path+'/*/*Localizations' + suffix):
            pc = o3d.io.read_point_cloud(file, format='xyz')
            down_pc = pc.uniform_down_sample(factor)
            downsampled_data = pd.DataFrame(np.asarray(down_pc.points), columns=['x', 'y', 'z'])
            downsampled_data.to_csv(file.split('.')[0] + '.csv', index=False)
        self.downsampled = True

    def denoise(self, min_cluster_size=10, min_samples=5, outlier_threshold=0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method for denoising the data using hdbscan. Overwrites the data in the preprocessed data folder.
        :param min_cluster_size: the desired minimum size of the cluster. Can be application-dependent
        :param min_samples: the desired minimum of samples in a cluster
        :param outlier_threshold: the threshold for the quantile for separating outliers from data
        :return: denoised data in the form of pandas DataFrame, outliers
        """
        if not self.downsampled:
            suffix = '.txt'
        else:
            suffix = '.csv'
        for file in glob.glob(self.train_path+'/*/*Localizations' + suffix):
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True,
                                        algorithm='best', alpha=0.7, metric='euclidean')
            data = pd.read_csv(file)
            clusterer.fit(data)
            threshold = pd.Series(clusterer.outlier_scores_).quantile(outlier_threshold)
            outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
            denoised_data = data.drop(data.index[outliers])
            #downsampled_data = pd.DataFrame(np.asarray(down_pc.points), columns=['x', 'y', 'z'])
            denoised_data.to_csv(file.split('.')[0] + '.csv', index=False)
        return denoised_data, outliers

    # def get_graphs(self, n_neighbors: int) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Method to get data in form of graphs
    #     :param n_neighbors: number of neighbors to be considered when building the graph
    #     :return: graph as array, graph as kneighbors graph
    #     """
    #     transformer = KNeighborsTransformer(n_neighbors=n_neighbors, algorithm='kd_tree')
    #     transformer.fit_transform(self.denoised_data)
    #     self.graph = transformer.kneighbors_graph()
    #     self.graph_array = self.graph.toarray()
    #     return self.graph_array, self.graph

