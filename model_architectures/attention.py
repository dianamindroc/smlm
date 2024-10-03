from torch import nn


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
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x.permute(1, 2, 0)  # (B, C, N)
