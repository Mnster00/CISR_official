import torch
import torch.nn as nn


class DegradationEncoder(nn.Module):
    """
    Lightweight degradation encoder E_d.
    5 strided 3x3 convolutional blocks with GroupNorm and LeakyReLU,
    followed by Global Average Pooling and a 1x1 projection,
    then variational heads to produce (mu_d, log_var_d).
    """

    def __init__(self, in_channels=3, latent_dim=256, num_groups=32):
        super().__init__()
        self.latent_dim = latent_dim

        channels = [64, 128, 256, 512, 512]

        layers = []
        in_ch = in_channels
        for out_ch in channels:
            layers.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
            )
            layers.append(nn.GroupNorm(num_groups, out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)

        self.projection = nn.Conv2d(channels[-1], latent_dim, kernel_size=1)

        self.variational_head_mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
        )
        self.variational_head_logvar = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        feat = self.conv_blocks(x)
        feat = self.projection(feat)
        feat = feat.mean(dim=[2, 3])

        mu_d = self.variational_head_mu(feat)
        log_var_d = self.variational_head_logvar(feat)

        return mu_d, log_var_d

    def sample(self, mu, log_var, deterministic=False):
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
