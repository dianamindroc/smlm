import torch
import torch.nn as nn


class PCNDecoderOnly(nn.Module):
    def __init__(self, original_model, num_dense=16384, latent_dim=1024, grid_size=4):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        # Copy the decoder parts from the original PCN model
        self.mlp = original_model.mlp
        self.final_conv = original_model.final_conv
        self.folding_seed = original_model.folding_seed

    def forward(self, encoded_features):
        B = encoded_features.shape[0]  # batch size
        coarse = self.mlp(encoded_features).reshape(-1, self.num_coarse, 3)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)
        seed = seed.reshape(B, -1, self.num_dense)

        feature_global = encoded_features.unsqueeze(2).expand(-1, -1, self.num_dense)
        feat = torch.cat([feature_global, seed, point_feat], dim=1)

        fine = self.final_conv(feat) + point_feat
        return coarse.contiguous(), fine.transpose(1, 2).contiguous()
