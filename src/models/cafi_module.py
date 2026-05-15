import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CAFIModule(nn.Module):
    """
    Causality-Aware Feature Interaction (CAFI) module.
    Uses cross-attention with learnable causal gates to enable
    controlled information flow between content and degradation
    branches while preserving causal independence.

    z_fused = z_c + s * G(Attn(z_c, z_d_expanded))
    where s = sigmoid(W_s [z_c; z_d]) is the causal strength score.
    """

    def __init__(self, content_dim=256, degradation_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.content_dim = content_dim
        self.degradation_dim = degradation_dim
        self.num_heads = num_heads
        self.d_k = content_dim // num_heads

        assert content_dim % num_heads == 0, \
            f"content_dim ({content_dim}) must be divisible by num_heads ({num_heads})"

        self.W_Q = nn.Linear(content_dim, content_dim)
        self.W_K = nn.Linear(degradation_dim, content_dim)
        self.W_V = nn.Linear(degradation_dim, content_dim)

        self.position_expansion = nn.Sequential(
            nn.Linear(degradation_dim, degradation_dim),
            nn.GELU(),
            nn.Linear(degradation_dim, degradation_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(content_dim, content_dim),
            nn.Sigmoid(),
        )

        self.causal_strength = nn.Sequential(
            nn.Linear(content_dim + degradation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.output_proj = nn.Linear(content_dim, content_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(content_dim)

    def expand_degradation(self, z_d, num_positions):
        z_d_expanded = z_d.unsqueeze(1).expand(-1, num_positions, -1)
        pe = self.position_expansion(z_d_expanded)
        return z_d_expanded + pe

    def forward(self, z_c, z_d):
        if z_c.dim() == 2:
            z_c = z_c.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        B, N, _ = z_c.shape

        z_d_expanded = self.expand_degradation(z_d, N)

        Q = self.W_Q(z_c)
        K = self.W_K(z_d_expanded)
        V = self.W_V(z_d_expanded)

        Q = Q.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.content_dim)
        attn_output = self.output_proj(attn_output)

        gated_output = self.gate(attn_output) * attn_output

        z_c_flat = z_c.mean(dim=1) if N > 1 else z_c.squeeze(1)
        z_d_flat = z_d
        s = self.causal_strength(torch.cat([z_c_flat, z_d_flat], dim=-1))
        s = s.unsqueeze(1)

        z_fused = z_c + s * gated_output
        z_fused = self.norm(z_fused)

        if squeeze_output:
            z_fused = z_fused.squeeze(1)

        return z_fused
