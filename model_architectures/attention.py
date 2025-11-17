from torch import nn
import torch.nn.functional as F
import torch
import math


# Self-attention module
class EnhancedSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(EnhancedSelfAttention, self).__init__()
        self.mha = nn.MultiheadAttention(in_channels, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels)
        )
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def forward(self, x):
        x = x.permute(2, 0, 1)  # (N, B, C)
        attn_output, attn_weights = self.mha(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x.permute(1, 2, 0), attn_weights  # (B, C, N)


class AttentionAverager(nn.Module):
    def __init__(self, embedding_dim, attention_dim=256):
        super().__init__()
        # Linear projections for attention mechanism
        self.linear_q = nn.Linear(embedding_dim, attention_dim)
        self.linear_k = nn.Linear(embedding_dim, attention_dim)
        self.linear_v = nn.Linear(embedding_dim, embedding_dim)  # Value dimension typically matches output dimension

    def forward(self, embeddings_list):
        if isinstance(embeddings_list, torch.Tensor):
            # If already a tensor, print shape and use directly
            batch = embeddings_list
        elif isinstance(embeddings_list, list):
            # If a list of tensors, stack them
            batch = torch.stack(embeddings_list)
        else:
            raise TypeError(f"Expected tensor or list of tensors, got {type(embeddings_list)}")

        # Project to Query, Key, Value
        query = self.linear_q(batch)
        key = self.linear_k(batch)
        value = self.linear_v(batch)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1))
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention
        attended_features = torch.matmul(attention_weights, value)

        # Aggregate across samples
        averaged_embedding = attended_features.mean(dim=0)

        return averaged_embedding


class RotationInvariantAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(RotationInvariantAttention, self).__init__()
        self.in_channels = in_channels

        # Original MHA mechanism
        self.mha = nn.MultiheadAttention(in_channels, num_heads)

        # Lightweight projection for invariant features
        self.invariant_proj = nn.Conv1d(3, in_channels, 1)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Linear(in_channels * 2, in_channels)
        )

        # Normalization
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

    def get_light_invariant_features(self, x):
        # x shape: (B, C, N)

        # Calculate centroid (invariant to rotation)
        centroid = torch.mean(x, dim=2, keepdim=True)  # (B, C, 1)

        # Calculate basic statistics that are rotation invariant
        # 1. Distance from each point to centroid
        dist_to_centroid = torch.norm(x - centroid, dim=1, keepdim=True)  # (B, 1, N)

        # 2. Standard deviation along channels (invariant to rotation)
        std_dev = torch.std(x, dim=1, keepdim=True)  # (B, 1, N)

        # 3. Average feature magnitude (also rotation invariant)
        feature_magnitude = torch.norm(x, dim=1, keepdim=True)  # (B, 1, N)

        # Concatenate these simple invariant features
        invariant_features = torch.cat([dist_to_centroid, std_dev, feature_magnitude], dim=1)  # (B, 3, N)

        # Project to original dimension
        return self.invariant_proj(invariant_features)  # (B, C, N)

    def forward(self, x):
        # x shape: (B, C, N)

        # Get lightweight rotation-invariant features
        inv_features = self.get_light_invariant_features(x)

        # Handle any potential NaNs
        inv_features = torch.nan_to_num(inv_features, nan=0.0)

        # Blend with original features
        x = x + 0.2 * inv_features

        # Apply attention
        x = x.permute(2, 0, 1)  # (N, B, C)
        attn_output, attn_weights = self.mha(x, x, x)

        # First residual
        x = self.norm1(x + attn_output)

        # Feed-forward
        ffn_output = self.ffn(x)

        # Second residual
        x = self.norm2(x + ffn_output)

        return x.permute(1, 2, 0), attn_weights