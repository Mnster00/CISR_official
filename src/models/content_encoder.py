import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentEncoder(nn.Module):
    """
    Content encoder E_c based on DINOv2-ViT-S/14 backbone.
    Extracts multi-scale features from layers {L/4, L/2, 3L/4, L}
    and aggregates them with learnable weights, then projects via
    a variational head to produce (mu_c, log_var_c) for the content
    latent distribution q(z_c | x).
    """

    def __init__(
        self,
        backbone_name="dinov2_vits14",
        latent_dim=256,
        freeze_backbone=True,
        selected_layers=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        self.backbone.eval()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        embed_dim = self.backbone.embed_dim
        num_layers = self.backbone.n_blocks

        if selected_layers is None:
            selected_layers = [
                num_layers // 4,
                num_layers // 2,
                3 * num_layers // 4,
                num_layers - 1,
            ]
        self.selected_layers = selected_layers

        self.projection_heads = nn.ModuleList(
            [nn.Linear(embed_dim, latent_dim) for _ in selected_layers]
        )

        num_selected = len(selected_layers)
        self.aggregation_weights = nn.Parameter(
            torch.ones(num_selected) / num_selected
        )

        self.variational_head_mu = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.variational_head_logvar = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        self._register_hooks()

    def _register_hooks(self):
        self.intermediate_features = {}
        for layer_idx in self.selected_layers:
            block = self.backbone.blocks[layer_idx]
            block.register_forward_hook(
                self._make_hook(layer_idx)
            )

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            self.intermediate_features[layer_idx] = output
        return hook

    def forward(self, x):
        B = x.shape[0]

        self.intermediate_features = {}
        _ = self.backbone(x)

        weights = F.softmax(self.aggregation_weights, dim=0)
        aggregated = torch.zeros(B, self.latent_dim, device=x.device)

        for i, layer_idx in enumerate(self.selected_layers):
            feat = self.intermediate_features[layer_idx]
            cls_token = feat[:, 0, :]
            projected = self.projection_heads[i](cls_token)
            aggregated = aggregated + weights[i] * projected

        mu_c = self.variational_head_mu(aggregated)
        log_var_c = self.variational_head_logvar(aggregated)

        return mu_c, log_var_c

    def sample(self, mu, log_var, deterministic=False):
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
