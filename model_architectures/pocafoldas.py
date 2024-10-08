import torch.nn as nn
import torch
from torch.nn import functional as F

from model_architectures.attention import EnhancedSelfAttention
from model_architectures.adaptive_folding import AdaptiveFolding


class Encoder(nn.Module):
    def __init__(self, latent_dim=1024, channels=3):
        super().__init__()

        self.latent_dim = latent_dim
        self.channels = channels

        self.first_conv = nn.Sequential(
            nn.Conv1d(self.channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.attention = EnhancedSelfAttention(self.latent_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, xyz):
        B, _, N = xyz.shape

        # Encoder block
        feature = self.first_conv(xyz)  # (B, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B, 512, N)
        feature = self.second_conv(feature)  # (B, 1024, N)

        feature = self.dropout(feature)

        # Attention
        attention_output = self.attention(feature)
        feature_global_return = torch.max(attention_output, dim=2, keepdim=False)[0]  # (B, 1024)

        return feature_global_return

class Decoder(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4, channels=3):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.channels = channels

        assert self.num_dense % self.grid_size ** 2 == 0
        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.channels * self.num_coarse)
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + self.channels + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.channels, 1)
        )

        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

        self.adaptive_folding = AdaptiveFolding(self.latent_dim)

    def forward(self, feature_global):
        B = feature_global.shape[0]

        # Decoder block
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, self.channels)  # (B, num_coarse, 3)
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, self.channels).transpose(2, 1)  # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_fine)
        feature_global_expanded = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)  # (B, 1024, num_fine)

        folded_points = self.adaptive_folding(feature_global_expanded, seed)
        feat = torch.cat([feature_global_expanded, seed, folded_points], dim=1)

        fine = self.final_conv(feat) + point_feat

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()


class Pocafoldas(nn.Module):
    def __init__(self, encoder=None, decoder=None, num_classes=2, classifier=False):
        super().__init__()

        self.encoder = encoder if encoder is not None else Encoder()
        self.decoder = decoder if decoder is not None else Decoder()
        self.classifier = classifier

        if self.classifier:
            self.mlp1 = nn.Linear(self.encoder.latent_dim, int(self.encoder.latent_dim / 2))
            self.bn1_mlp = nn.BatchNorm1d(int(self.encoder.latent_dim / 2))
            self.mlp2 = nn.Linear(int(self.encoder.latent_dim / 2), int(self.encoder.latent_dim / 4))
            self.bn2_mlp = nn.BatchNorm1d(int(self.encoder.latent_dim / 4))
            self.mlp3 = nn.Linear(int(self.encoder.latent_dim / 4), num_classes)
            self.sigmoid_mlp = nn.Sigmoid()

    def mlp_classification(self, z):
        if self.classifier:
            out_mlp = F.relu(self.bn1_mlp(self.mlp1(z)))
            out_mlp = F.relu(self.bn2_mlp(self.mlp2(out_mlp)))
            out_mlp = self.mlp3(out_mlp)
            return out_mlp
        else:
            return None

    def forward(self, xyz):
        feature_global = self.encoder(xyz)

        if self.classifier:
            out_classifier = self.mlp_classification(feature_global)

        coarse, fine = self.decoder(feature_global)

        if self.classifier:
            return coarse, fine, feature_global, out_classifier
        else:
            return coarse, fine, feature_global
