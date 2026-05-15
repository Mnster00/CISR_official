import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    L_rec = ||x - x_hat_LR||_1
    L_SR = ||y - y_hat_HR||_1
    """

    def __init__(self, loss_type="l1"):
        super().__init__()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, pred, target):
        return self.loss_fn(pred, target)


class KLDivergenceLoss(nn.Module):
    """
    KL divergence between encoder posteriors q(z|x) and standard normal prior N(0, I).
    L_KL = D_KL(q(z_c|x) || p(z_c)) + D_KL(q(z_d|x) || p(z_d))

    For diagonal Gaussian: D_KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    """

    def __init__(self):
        super().__init__()

    def forward(self, mu, log_var):
        kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
        return kl.mean()


class InfoNCELoss(nn.Module):
    """
    Interventional Contrastive Loss (InfoNCE).

    L_NCE^c = -E[log(exp(sim(z_c^ref, z_c^pos) / tau) / sum_k(exp(sim(z_c^ref, z_c^k) / tau)))]

    Content contrastive: positives share same content but different degradation.
    Degradation contrastive: positives share same degradation but different content.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_ref, z_pos, z_neg):
        z_ref = F.normalize(z_ref, dim=-1)
        z_pos = F.normalize(z_pos, dim=-1)
        z_neg = F.normalize(z_neg, dim=-1)

        pos_sim = torch.sum(z_ref * z_pos, dim=-1) / self.temperature

        if z_neg.dim() == 2:
            neg_sim = torch.matmul(z_ref, z_neg.t()) / self.temperature
        else:
            neg_sim = torch.bmm(
                z_ref.unsqueeze(1),
                z_neg.transpose(1, 2)
            ).squeeze(1) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class PriorRegularization(nn.Module):
    """
    L_prior = ||z_clean||_2^2
    Regularizes the learnable clean prior to remain within
    the high-density region of p(d) = N(0, I).
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_clean):
        return torch.norm(z_clean, p=2).pow(2)


class HingeAdversarialLoss(nn.Module):
    """
    Hinge loss for adversarial training.
    L_adv = max(0, 1 - D(y_real)) for discriminator
    L_adv = -D(y_fake) for generator
    """

    def __init__(self):
        super().__init__()

    def discriminator_loss(self, d_real, d_fake):
        loss_real = F.relu(1.0 - d_real).mean()
        loss_fake = F.relu(1.0 + d_fake).mean()
        return loss_real + loss_fake

    def generator_loss(self, d_fake):
        return -d_fake.mean()


class CISRLoss(nn.Module):
    """
    Total loss for CISR training:
    L_total = lambda_rec * L_rec + lambda_SR * L_SR + lambda_KL * L_KL
              + lambda_NCE * (L_NCE^c + L_NCE^d) + lambda_adv * L_adv
              + lambda_prior * L_prior
    """

    def __init__(
        self,
        lambda_rec=1.0,
        lambda_sr=1.0,
        lambda_kl=0.01,
        lambda_nce=0.1,
        lambda_adv=0.1,
        lambda_prior=0.001,
        temperature=0.07,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_sr = lambda_sr
        self.lambda_kl = lambda_kl
        self.lambda_nce = lambda_nce
        self.lambda_adv = lambda_adv
        self.lambda_prior = lambda_prior

        self.rec_loss = ReconstructionLoss("l1")
        self.kl_loss = KLDivergenceLoss()
        self.nce_loss = InfoNCELoss(temperature=temperature)
        self.prior_loss = PriorRegularization()
        self.adv_loss = HingeAdversarialLoss()

    def forward(self, model_output, batch, d_real=None, d_fake=None):
        x_lr = batch["lr_ref"]
        y_hr = batch["hr"]

        x_hat_lr = model_output["x_hat_lr"]
        y_hat_hr = model_output["y_hat_hr"]

        l_rec = self.rec_loss(x_hat_lr, x_lr)
        l_sr = self.rec_loss(y_hat_hr, y_hr)

        l_kl_c = self.kl_loss(model_output["mu_c"], model_output["log_var_c"])
        l_kl_d = self.kl_loss(model_output["mu_d"], model_output["log_var_d"])
        l_kl = l_kl_c + l_kl_d

        l_prior = self.prior_loss(model_output["z_clean"])

        l_nce_c = torch.tensor(0.0, device=x_lr.device)
        l_nce_d = torch.tensor(0.0, device=x_lr.device)

        if "lr_pos_c" in batch and "lr_neg_c" in batch:
            z_c_ref = model_output["z_c"]
            z_c_pos, _, _ = self._encode_content(batch["lr_pos_c"])
            z_c_neg, _, _ = self._encode_content(batch["lr_neg_c"])
            l_nce_c = self.nce_loss(z_c_ref, z_c_pos, z_c_neg)

        if "lr_pos_d" in batch and "lr_neg_d" in batch:
            z_d_ref = model_output["z_d"]
            z_d_pos, _, _ = self._encode_degradation(batch["lr_pos_d"])
            z_d_neg, _, _ = self._encode_degradation(batch["lr_neg_d"])
            l_nce_d = self.nce_loss(z_d_ref, z_d_pos, z_d_neg)

        l_adv = torch.tensor(0.0, device=x_lr.device)
        if d_fake is not None:
            l_adv = self.adv_loss.generator_loss(d_fake)

        l_total = (
            self.lambda_rec * l_rec
            + self.lambda_sr * l_sr
            + self.lambda_kl * l_kl
            + self.lambda_nce * (l_nce_c + l_nce_d)
            + self.lambda_adv * l_adv
            + self.lambda_prior * l_prior
        )

        loss_dict = {
            "l_rec": l_rec.item(),
            "l_sr": l_sr.item(),
            "l_kl": l_kl.item(),
            "l_kl_c": l_kl_c.item(),
            "l_kl_d": l_kl_d.item(),
            "l_nce_c": l_nce_c.item() if isinstance(l_nce_c, torch.Tensor) else l_nce_c,
            "l_nce_d": l_nce_d.item() if isinstance(l_nce_d, torch.Tensor) else l_nce_d,
            "l_adv": l_adv.item() if isinstance(l_adv, torch.Tensor) else l_adv,
            "l_prior": l_prior.item(),
            "l_total": l_total.item(),
        }

        return l_total, loss_dict

    def _encode_content(self, x):
        return None, None, None

    def _encode_degradation(self, x):
        return None, None, None
