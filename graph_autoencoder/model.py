
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.nn import GAE
import torch

from helpers.config import Config

# from .dgcnn.pytorch.model import PointNet

class GCNEncoder(torch.nn.Module):
    """
    Simple GCN encoder. Inspired by Antonio Longa tutorial series.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True) # cached only for transductive learning

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)


class GAEModel(torch.nn.Module):
    def __init__(self, cfg: Config):
        super(GAEModel, self).__init__()
        self.config = Config.from_json(cfg)
        if self.config.model.encoder == 'GCNConv':
            self.encoder = GCNEncoder(self.config.model.nr_features, self.config.model.output)
        else:
            raise NotImplementedError('Encoder not implemented.')
        if self.config.train.optimizer.type == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train.optimizer.lr)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decoder = GAE(self.encoder).to(self.device)
        #self.load_data()

    def load_data(self):
        """
        Loads (already preprocessed) data from path specified in config. Expects .pt format.
        Splits data (edges) into train and test.
        :return: None
        """
        self.data = torch.load(self.config.data.path)
        self.data_ = train_test_split_edges(self.data)
        self.train_pos_edge_index = self.data.train_pos_edge_index.to(self.device)
    def train(self):
        """
        Trains the model.
        :return: z: Latent variable
        """
        self.optimizer.zero_grad()
        z = self.encoder.encode(self.data_, self.train_pos_edge_index)
        loss = self.decoder.recon_loss(z, self.train_pos_edge_index)
        loss.backward()
        self.optimizer.step()
        return float(loss)
    def evaluate(self, pos_edge_index, neg_edge_index):
        """
        Tests the model.
        :param z: Latent variable
        :param pos_edge_index: Positive edges from train_test_split_edges
        :param neg_edge_index: Negative edges from train_test_split_edges
        :return: Accuracy of the model
        """
        self.decoder.eval()
        with torch.no_grad():
            z = self.encoder.encode(self.data_, self.train_pos_edge_index)
        return self.decoder.test(z, pos_edge_index, neg_edge_index)