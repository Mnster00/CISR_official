import torch
import torch.nn as nn

from .content_encoder import ContentEncoder
from .degradation_encoder import DegradationEncoder
from .cafi_module import CAFIModule
from .decoder import CausalGatedDecoder
from .discriminator import UNetDiscriminator


class CISRModel(nn.Module):
    """
    Causal Intervention for Super-Resolution (CISR) model.

    Integrates:
    - Content encoder E_c (DINOv2 backbone + variational head)
    - Degradation encoder E_d (lightweight CNN + variational head)
    - CAFI module (cross-attention with causal gates)
    - Causal-Gated Decoder G_theta (ResBlocks + AdaIN)
    - Learnable clean prior z_clean (initialized to zero)
    - U-Net Discriminator with spectral normalization
    """

    def __init__(
        self,
        backbone_name="dinov2_vits14",
        latent_dim=256,
        degradation_latent_dim=256,
        decoder_num_features=128,
        decoder_num_res_blocks=16,
        scale_factor=4,
        freeze_backbone=True,
        num_heads=8,
        cafi_dropout=0.1,
    ):
        super().__init__()

        self.content_encoder = ContentEncoder(
            backbone_name=backbone_name,
            latent_dim=latent_dim,
            freeze_backbone=freeze_backbone,
        )

        self.degradation_encoder = DegradationEncoder(
            in_channels=3,
            latent_dim=degradation_latent_dim,
        )

        self.cafi = CAFIModule(
            content_dim=latent_dim,
            degradation_dim=degradation_latent_dim,
            num_heads=num_heads,
            dropout=cafi_dropout,
        )

        self.decoder = CausalGatedDecoder(
            latent_dim=latent_dim,
            num_features=decoder_num_features,
            num_res_blocks=decoder_num_res_blocks,
            degradation_dim=degradation_latent_dim,
            scale_factor=scale_factor,
            out_channels=3,
        )

        self.discriminator = UNetDiscriminator(
            in_channels=3,
            base_channels=64,
            num_layers=7,
        )

        self.z_clean = nn.Parameter(torch.zeros(degradation_latent_dim))

        self.scale_factor = scale_factor
        self.latent_dim = latent_dim
        self.degradation_latent_dim = degradation_latent_dim

    def encode_content(self, x, deterministic=False):
        mu_c, log_var_c = self.content_encoder(x)
        z_c = self.content_encoder.sample(mu_c, log_var_c, deterministic)
        return z_c, mu_c, log_var_c

    def encode_degradation(self, x, deterministic=False):
        mu_d, log_var_d = self.degradation_encoder(x)
        z_d = self.degradation_encoder.sample(mu_d, log_var_d, deterministic)
        return z_d, mu_d, log_var_d

    def decode(self, z_fused, z_d, input_size=None):
        return self.decoder(z_fused, z_d, input_size)

    def forward(self, x_lr, deterministic=False):
        z_c, mu_c, log_var_c = self.encode_content(x_lr, deterministic)
        z_d, mu_d, log_var_d = self.encode_degradation(x_lr, deterministic)

        z_fused_lr = self.cafi(z_c, z_d)
        x_hat_lr = self.decode(z_fused_lr, z_d, input_size=x_lr.shape[2:])

        z_fused_hr = self.cafi(z_c, self.z_clean)
        lr_h, lr_w = x_lr.shape[2], x_lr.shape[3]
        hr_h, hr_w = lr_h * self.scale_factor, lr_w * self.scale_factor
        y_hat_hr = self.decode(z_fused_hr, self.z_clean, input_size=(lr_h, lr_w))

        return {
            "z_c": z_c,
            "z_d": z_d,
            "mu_c": mu_c,
            "log_var_c": log_var_c,
            "mu_d": mu_d,
            "log_var_d": log_var_d,
            "z_fused_lr": z_fused_lr,
            "z_fused_hr": z_fused_hr,
            "x_hat_lr": x_hat_lr,
            "y_hat_hr": y_hat_hr,
            "z_clean": self.z_clean,
        }

    def inference(self, x_lr):
        self.eval()
        with torch.no_grad():
            z_c, _, _ = self.encode_content(x_lr, deterministic=True)
            z_fused = self.cafi(z_c, self.z_clean)
            lr_h, lr_w = x_lr.shape[2], x_lr.shape[3]
            y_hat_hr = self.decode(z_fused, self.z_clean, input_size=(lr_h, lr_w))
        return y_hat_hr

    def get_discriminator_loss(self, y_real, y_fake):
        d_real = self.discriminator(y_real)
        d_fake = self.discriminator(y_fake.detach())

        loss_real = torch.mean(F.relu(1.0 - d_real))
        loss_fake = torch.mean(F.relu(1.0 + d_fake))
        d_loss = loss_real + loss_fake
        return d_loss

    def get_generator_adv_loss(self, y_fake):
        d_fake = self.discriminator(y_fake)
        g_loss = -torch.mean(d_fake)
        return g_loss
