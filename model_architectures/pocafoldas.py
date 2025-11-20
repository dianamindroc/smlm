# PCN network inspired from: Copyright <2023> <https://github.com/qinglew/PCN-PyTorch>
# Adapted by adding self-attention, dropout and adaptive folding

import torch
import torch.nn as nn
from torch.nn import functional as F

from model_architectures.attention import EnhancedSelfAttention, RotationInvariantAttention
from model_architectures.adaptive_folding import AdaptiveFolding

class PocaFoldAS(nn.Module):
    """
    PocaFoldAS: adapted Point Cloud Completion Network (PCN-inspired).

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=2048, latent_dim=1024, grid_size=2, classifier=False, num_classes=2, channels=3, decoder_type="folding"):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.classifier = classifier
        self.num_classes = num_classes
        self.channels = channels

        #assert self.num_dense % self.grid_size ** 2 == 0
        #self.num_coarse = self.num_dense // (self.grid_size ** 2)

        grid_patch = self.grid_size ** 2

        # Adjust num_dense to be divisible by grid_size^2
        if self.num_dense % grid_patch != 0:
            original_num_dense = self.num_dense
            self.num_dense = ((self.num_dense // grid_patch) + 1) * grid_patch
            print(
                f"[INFO] Adjusted num_dense from {original_num_dense} to {self.num_dense} to match folding grid size ({self.grid_size}x{self.grid_size})")

        self.num_coarse = self.num_dense // grid_patch


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

        ## original folding_seed
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(
            self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)
        ## adapted folding seed
        #base_x = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float)
        #base_y = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float)

        # Add small random perturbations
        #noise_x = torch.randn_like(base_x) * 0.01
        #noise_y = torch.randn_like(base_y) * 0.01

        #base_x = base_x + noise_x
        #base_y = base_y + noise_y

        #grid_x, grid_y = torch.meshgrid(base_x, base_y)
        #grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)
        #self.folding_seed = grid.transpose(1, 2).cuda()

        self.output_scale = nn.Parameter(torch.ones(channels), requires_grad=True)

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

        # try out attention
        attention_output, attn_weights = self.attention(feature)
        feature = self.dropout(attention_output)
        # Skip connection for attention
        #attention_output = attention_output #+ encoder_skip

        #feature_global_return = torch.max(feature, dim=2, keepdim=False)[0]
        feature_global_return = torch.mean(feature, dim=2, keepdim=False) # (B, 1024)

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
        feature_global = feature_global_return.unsqueeze(2).expand(-1, -1, self.num_dense)  # (B, 1024, num_fine)

        # TODO: added this check if good
        # attended_features = self.folding_attention(feature_global, seed)
        # attended_features = attended_features.expand(-1, -1, self.num_dense)
        # skip = torch.cat([feature_global, feature_global1, seed, point_feat], dim=1)

        # feat = torch.cat([feature_global, seed, point_feat], dim=1)  # (B, 1024+2+3, num_fine)
        folded_points = self.adaptive_folding(feature_global, seed)
        feat = torch.cat([feature_global, seed, folded_points], dim=1)
        refinement_input = feat

        # Skip connection for final_conv
        #skip_out = self.skip_conv(feat)
        if self.channels == 3:
            fine = self.final_conv(feat) + point_feat #* self.output_scale.view(1, -1, 1)
        else:
            fine = self.final_conv(refinement_input)
            fine_xyz = fine[:, :3, :] + point_feat[:, :3, :]
            fine_sigma = fine[:, 3:, :]
            coarse_xyz = coarse[:, :, :3]
            coarse_sigma = coarse[:, :, 3:]

        if self.channels == 3:
            if self.classifier:
                return coarse.contiguous(), fine.transpose(1, 2).contiguous(), feature_global_return, out_classifier, attn_weights
            else:
                return coarse.contiguous(), fine.transpose(1, 2).contiguous(), feature_global_return, attn_weights
        else:
            fine_xyz = fine[:, :3, :].transpose(2, 1).contiguous()
            fine_sigma = fine[:, 3:, :].transpose(2, 1).contiguous()
            coarse_xyz = coarse[:, :, :3]
            coarse_sigma = coarse[:, :, 3:]
            if self.classifier:
                return (coarse_xyz.contiguous(), coarse_sigma.contiguous()), (fine_xyz, fine_sigma), feature_global_return, out_classifier, attn_weights
            else:
                return (coarse_xyz.contiguous(), coarse_sigma.contiguous()), (fine_xyz, fine_sigma), feature_global_return, attn_weights


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
        self.decoder_type = decoder_type
    def encode_latent(self, xyz):
        B, _, N = xyz.shape
        feature = self.first_conv(xyz)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)
        feature = self.second_conv(feature)
        attention_output, attn_weights = self.attention(feature)
        feature = self.dropout(attention_output)
        feature_global_return = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global_return, attn_weights
