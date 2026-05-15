import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaINModulation(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) modulation.
    h_{l+1} = gamma(z_d) * (h_l - mu(h_l)) / sigma(h_l) + beta(z_d)
    """

    def __init__(self, num_features, degradation_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.gamma_fc = nn.Linear(degradation_dim, num_features)
        self.beta_fc = nn.Linear(degradation_dim, num_features)

    def forward(self, h, z_d):
        h_norm = self.norm(h)
        gamma = self.gamma_fc(z_d).unsqueeze(2).unsqueeze(3)
        beta = self.beta_fc(z_d).unsqueeze(2).unsqueeze(3)
        return gamma * h_norm + beta


class ResBlockWithAdaIN(nn.Module):
    """Residual block with AdaIN modulation from degradation code z_d."""

    def __init__(self, num_features, degradation_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.adain1 = AdaINModulation(num_features, degradation_dim)
        self.adain2 = AdaINModulation(num_features, degradation_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, h, z_d):
        residual = h
        h = self.conv1(h)
        h = self.adain1(h, z_d)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.adain2(h, z_d)
        return h + residual


class ChannelAttention(nn.Module):
    """Channel attention mechanism for dynamic feature modulation."""

    def __init__(self, num_features, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_features // reduction, num_features, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(b, c, 1, 1)


class CausalGatedDecoder(nn.Module):
    """
    Causal-Gated Decoder G_theta.
    16 residual blocks with AdaIN modulation from z_d,
    channel attention, and pixel-shuffle upsampling.

    For LR reconstruction: x_hat_LR = G(z_fused_LR, z_d)
    For HR reconstruction: y_hat_HR = G(z_fused_HR, z_clean)
    """

    def __init__(
        self,
        latent_dim=256,
        num_features=128,
        num_res_blocks=16,
        degradation_dim=256,
        scale_factor=4,
        out_channels=3,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.scale_factor = scale_factor

        self.input_proj = nn.Conv2d(latent_dim, num_features, 3, 1, 1)

        self.res_blocks = nn.ModuleList(
            [ResBlockWithAdaIN(num_features, degradation_dim) for _ in range(num_res_blocks)]
        )

        self.channel_attention = ChannelAttention(num_features)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if scale_factor == 4:
            self.upsample2 = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif scale_factor == 8:
            self.upsample2 = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.upsample3 = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.output_proj = nn.Conv2d(num_features, out_channels, 3, 1, 1)

    def forward(self, z_fused, z_d, input_size=None):
        if input_size is None:
            h = 48
            w = 48
        else:
            h, w = input_size

        x = z_fused.unsqueeze(-1).unsqueeze(-1)
        x = x.expand(-1, -1, h, w)

        x = self.input_proj(x)

        for res_block in self.res_blocks:
            x = res_block(x, z_d)

        x = self.channel_attention(x)

        x = self.upsample(x)
        if self.scale_factor >= 4:
            x = self.upsample2(x)
        if self.scale_factor >= 8:
            x = self.upsample3(x)

        x = self.output_proj(x)
        return x
