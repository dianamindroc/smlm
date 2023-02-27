from .train import train
from .train import test
from .model import GCNEncoder
from .dataset import SMLMDataset
from torch_geometric.nn import GAE

# TODO: has to be replaced, it is deprecated
from torch_geometric.transforms import RandomLinkSplit

# Model
model = GAE(GCNEncoder(num_features, out_channels))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.float()
model = model.to(device)
x = x.float()
x = data_split.x.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Data
data = torch.load('/home/dim26fa/data/imod_models/mitochondria/testing_dataset_class/processed/data_0.pt')
data_ = train_test_split_edges(data)
train_pos_edge_index = data.train_pos_edge_index.to(device)

# Training loop
for epoch in range(1, epochs + 1):
    loss = train()
    auc, ap = test(data_split.test_pos_edge_index, data_split.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))


