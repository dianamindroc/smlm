import os.path as osp
import torch
from torch_geometric.data import Dataset, download_url, Data
import glob
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsTransformer
import networkx as nx

class SMLMDataset(Dataset):
    """
    Create SMLMDataset class.
    """
    def __init__(self, root: str, filename: str, suffix: str, transform = None, pre_transform = None, pre_filter = None, test: bool = False):
        """
        :param root: path to where the data is stored; it is split into raw_dir and processed_dir
        """
        self.filenames = glob.glob(osp.join(root, filename) + suffix)
        self.test = test
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """
        If these files exist in raw_dir, the download is not triggered
        """
        return self.filenames

    @property
    def processed_file_names(self):
        """
        If these files exist in processed_dir, processing is skipped
        """
        return 'not_implemented_yet.pt'

    def download(self):
        pass

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            df = pd.read_csv(raw_path)
            transformer = KNeighborsTransformer(n_neighbors=15, algorithm='brute')
            transformer.fit_transform(df)
            graph = transformer.kneighbors_graph()
            nx_graph = nx.from_numpy_array(graph.toarray())
            edges = np.transpose(nx_graph.edges)
            data = Data(x=np.array(df),
                    edge_index=edges,
                    # edge_attr=edge_attr, TODO: add edge weights later
                    # y=label, TODO: add label later
                    )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data