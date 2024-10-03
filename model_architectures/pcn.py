# PCN network inspired from: Copyright <2023> <https://github.com/qinglew/PCN-PyTorch>
# Adapted by adding self-attention, dropout and adaptive folding

import torch
import torch.nn as nn
from torch.nn import functional as F

from model_architectures.attention import EnhancedSelfAttention
from model_architectures.adaptive_folding import AdaptiveFolding, OptimizedFoldingModule


class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4, classifier=False, num_classes=2, channels=3):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.classifier = classifier
        self.num_classes = num_classes
        self.channels = channels

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

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
        self.adaptive_folding = AdaptiveFolding(self.latent_dim)

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

        # Skip connection for final_conv
        self.skip_conv = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 3, 1)
        )
        # Add layer normalization
        self.group_norm = nn.GroupNorm(32, latent_dim)


        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(
            self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

        if self.classifier:
            #self.classifiernet = ImprovedClassifier(self.latent_dim, self.num_classes)
            #MLP for binary classification
            self.mlp1 = nn.Linear(self.latent_dim, int(self.latent_dim/2))
            self.bn1_mlp = nn.BatchNorm1d(int(self.latent_dim/2))
            self.mlp2 = nn.Linear(int(self.latent_dim/2), int(self.latent_dim/4))
            self.bn2_mlp = nn.BatchNorm1d(int(self.latent_dim/4))
            self.mlp3 = nn.Linear(int(self.latent_dim/4), self.num_classes)
            # self.bn3_mlp = nn.BatchNorm1d(1)
            self.sigmoid_mlp = nn.Sigmoid()

        self.dropout = nn.Dropout(0.5)

    def mlp_classification(self, z):  # icetin: mlp part that is connected to z
       if self.classifier:
           out_mlp = F.relu(self.bn1_mlp(self.mlp1(z)))  # input: z output: prediction
           out_mlp = F.relu(self.bn2_mlp(self.mlp2(out_mlp)))
           out_mlp = self.mlp3(out_mlp)
           #out_mlp = self.sigmoid_mlp(out_mlp)
           return out_mlp
       else:
           return None

    def forward(self, xyz):
        B, _, N = xyz.shape

        # encoder
        # feature = self.first_conv(xyz.transpose(2, 1))  # (B,  256, N)
        feature = self.first_conv(xyz)  # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
        feature = self.second_conv(feature)  # (B, 1024, N)

        # Skip connection for encoder
        #encoder_skip = 0.1 * feature
        feature = self.dropout(feature)
        # try out attention
        attention_output = self.attention(feature)

        # Skip connection for attention
        attention_output = attention_output #+ encoder_skip

        feature_global_return = torch.max(attention_output, dim=2, keepdim=False)[0]  # (B, 1024)

        # classifier
        if self.classifier:
            out_classifier = self.mlp_classification(feature_global_return)
            #out_classifier = self.classifiernet(feature_global_return)

        # decoder
        coarse = self.mlp(feature_global_return).reshape(-1, self.num_coarse, self.channels)

        # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, self.channels).transpose(2, 1)  # (B, 3, num_fine)

        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_fine)
        # seed = self.folding_seed.expand(B, -1, self.num_dense)
        feature_global = feature_global_return.unsqueeze(2).expand(-1, -1, self.num_dense)  # (B, 1024, num_fine)

        # TODO: added this check if good
        # attended_features = self.folding_attention(feature_global, seed)
        # attended_features = attended_features.expand(-1, -1, self.num_dense)
        # skip = torch.cat([feature_global, feature_global1, seed, point_feat], dim=1)

        # feat = torch.cat([feature_global, seed, point_feat], dim=1)  # (B, 1024+2+3, num_fine)
        folded_points = self.adaptive_folding(feature_global, seed)
        feat = torch.cat([feature_global, seed, folded_points], dim=1)

        # Skip connection for final_conv
        # skip_out = self.skip_conv(feat)
        if self.channels == 3:
            fine = self.final_conv(feat) + point_feat #+ 0.1 * skip_out
        else:
            fine = self.final_conv(feat)
            print('size of fine is ', fine.size())
            print('size of point_feat is ', point_feat.size())
            fine_xyz = fine[:, :3, :] + point_feat[:, :3, :]  # (B, 3, num_fine)
            fine_sigma = fine[:, 3:, :]
            coarse_xyz = coarse[:, :, :3]
            coarse_sigma = coarse[:, :, 3:]

        if self.channels == 3:
            if self.classifier:
                return coarse.contiguous(), fine.transpose(1, 2).contiguous(), feature_global_return, out_classifier
            else:
                return coarse.contiguous(), fine.transpose(1, 2).contiguous(), feature_global_return
        else:
            fine_xyz = fine_xyz.transpose(2, 1).contiguous()
            fine_sigma = fine_sigma.transpose(2, 1).contiguous()
            if self.classifier:
                return (coarse_xyz.contiguous(), coarse_sigma.contiguous()), (fine_xyz, fine_sigma), feature_global_return, out_classifier
            else:
                return (coarse_xyz.contiguous(), coarse_sigma.contiguous()), (fine_xyz, fine_sigma), feature_global_return


class ImprovedClassifier(nn.Module):
    def __init__(self, latent_dim, num_classes, dropout_rate=0.3):
        super(ImprovedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.BatchNorm1d(latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.BatchNorm1d(latent_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(latent_dim // 4, num_classes),
        )

    def forward(self, z):
        return self.classifier(z)