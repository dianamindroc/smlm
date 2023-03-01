from .config import cfg
from .model import GAEModel

# Training loop
for epoch in range(1, cfg['train']['epochs'] + 1):
    loss = GAEModel.train()
    auc, ap = GAEModel.evaluate(GAEModel.data_.test_pos_edge_index, GAEModel.data_.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
