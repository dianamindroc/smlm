from torch import nn
import torch

# Multiple folding mechanisms

class AdaptiveFolding(nn.Module):
    def __init__(self, feature_dim, grid_dim=2, hidden_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + grid_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, features, grid):
        B, C, N = features.shape
        features = features.permute(0, 2, 1).expand(-1, grid.shape[2], -1)
        grid = grid.permute(0, 2, 1).expand(B, -1, 2)
        folding_input = torch.cat([features, grid], dim=-1)
        return self.mlp(folding_input).permute(0, 2, 1)


class OptimizedFoldingModule(nn.Module):
    def __init__(self, feature_dim, grid_dim=2, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + grid_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, features, grid):
        # Avoid permute and expand, use broadcasting
        features = features.transpose(1, 2).unsqueeze(2)  # [B, N, 1, C]
        grid = grid.transpose(1, 2).unsqueeze(3)  # [B, N, 2, 1]

        # Use broadcasting instead of cat
        folding_input = torch.cat([features.expand(-1, -1, grid.shape[2], -1),
                                   grid.expand(-1, -1, -1, features.shape[3])], dim=3)

        # Reshape for MLP input
        folding_input = folding_input.view(-1, features.shape[3] + grid.shape[2])

        # Apply MLP and reshape output
        output = self.mlp(folding_input)
        return output.view(features.shape[0], -1, 3).transpose(1, 2)

class EnhancedAdaptiveFolding(nn.Module):
    def __init__(self, feature_dim, grid_dim=2, hidden_dim=512):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(feature_dim + grid_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(feature_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, features, grid):
        print(f"Features shape: {features.shape}")
        print(f"Grid shape: {grid.shape}")
        B, C, N = features.shape
        features = features.permute(0, 2, 1).expand(-1, grid.shape[2], -1)
        grid = grid.permute(0, 2, 1).expand(B, -1, 2)

        print(f"Reshaped features shape: {features.shape}")
        print(f"Reshaped grid shape: {grid.shape}")

        folding_input1 = torch.cat([features, grid], dim=-1)
        print(f"Folding input1 shape: {folding_input1.shape}")

        fold1 = self.mlp1(folding_input1)
        print(f"Fold1 shape: {fold1.shape}")

        folding_input2 = torch.cat([features, fold1], dim=-1)
        print(f"Folding input2 shape: {folding_input2.shape}")

        fold2 = self.mlp2(folding_input2)
        print(f"Fold2 shape: {fold2.shape}")

        return (fold1 + fold2).permute(0, 2, 1)  # Residual connection
