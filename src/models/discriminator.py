import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class UNetDiscriminator(nn.Module):
    """
    U-Net Discriminator with spectral normalization.
    Provides both global and local adversarial feedback.
    Receives 512x512 patches and produces per-pixel realness maps.
    """

    def __init__(self, in_channels=3, base_channels=64, num_layers=7):
        super().__init__()

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        ch = in_channels
        enc_channels = []

        for i in range(num_layers):
            out_ch = min(base_channels * (2 ** i), 512)
            self.encoder_layers.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(ch, out_ch, 4, 2, 1)),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            enc_channels.append(out_ch)
            ch = out_ch

        self.bottleneck = nn.Sequential(
            spectral_norm(nn.Conv2d(ch, ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(ch, ch, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        for i in range(num_layers - 1, -1, -1):
            in_ch = ch + enc_channels[i]
            out_ch = enc_channels[i] if i > 0 else base_channels
            self.decoder_layers.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1)),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            ch = out_ch

        self.output_layer = spectral_norm(nn.Conv2d(ch, 1, 3, 1, 1))

    def forward(self, x):
        enc_features = []
        for layer in self.encoder_layers:
            x = layer(x)
            enc_features.append(x)

        x = self.bottleneck(x)

        for i, layer in enumerate(self.decoder_layers):
            idx = len(self.decoder_layers) - 1 - i
            enc_feat = enc_features[idx]
            x = F.interpolate(x, size=enc_feat.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, enc_feat], dim=1)
            x = layer(x)

        out = self.output_layer(x)
        return out
