import os
import sys
import argparse
import logging
import math
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.models import CISRModel
from src.data import build_dataloader, DegradationPipeline
from src.losses import CISRLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train CISR model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="results/train",
                        help="Output directory for checkpoints and logs")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("CISR")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(output_dir, "train.log"))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def build_contrastive_batch(batch, model, device):
    lr_all = batch["lr_all"].to(device)
    B, N, C, H, W = lr_all.shape

    if N < 2:
        return None

    ref_idx = 0
    pos_c_idx = 1

    lr_ref = lr_all[:, ref_idx]
    lr_pos_c = lr_all[:, pos_c_idx]

    neg_indices = torch.randperm(B)
    while torch.any(neg_indices == torch.arange(B, device=neg_indices.device)):
        neg_indices = torch.randperm(B)
    lr_neg_c = lr_all[neg_indices, ref_idx]

    batch["lr_ref"] = lr_ref
    batch["lr_pos_c"] = lr_pos_c
    batch["lr_neg_c"] = lr_neg_c

    return batch


def train_one_iteration(model, batch, criterion, optimizer_G, optimizer_D, device):
    model.train()

    batch = build_contrastive_batch(batch, model, device)

    lr_ref = batch["lr_ref"].to(device)
    y_hr = batch["hr"].to(device)

    model_output = model(lr_ref, deterministic=False)

    y_hat_hr = model_output["y_hat_hr"]
    if y_hat_hr.shape[2:] != y_hr.shape[2:]:
        y_hat_hr = F.interpolate(y_hat_hr, size=y_hr.shape[2:], mode="bilinear", align_corners=False)

    d_real = model.discriminator(y_hr)
    d_fake = model.discriminator(y_hat_hr.detach())
    d_loss = criterion.adv_loss.discriminator_loss(d_real, d_fake)

    optimizer_D.zero_grad()
    d_loss.backward()
    optimizer_D.step()

    model_output = model(lr_ref, deterministic=False)
    y_hat_hr = model_output["y_hat_hr"]
    if y_hat_hr.shape[2:] != y_hr.shape[2:]:
        y_hat_hr = F.interpolate(y_hat_hr, size=y_hr.shape[2:], mode="bilinear", align_corners=False)

    d_fake_for_g = model.discriminator(y_hat_hr)

    z_c_ref = model_output["z_c"]
    z_c_pos, _, _ = model.encode_content(batch["lr_pos_c"].to(device))
    z_c_neg, _, _ = model.encode_content(batch["lr_neg_c"].to(device))

    l_nce_c = criterion.nce_loss(z_c_ref, z_c_pos, z_c_neg)

    z_d_ref = model_output["z_d"]
    z_d_pos, _, _ = model.encode_degradation(batch["lr_pos_c"].to(device))
    z_d_neg, _, _ = model.encode_degradation(batch["lr_neg_c"].to(device))
    l_nce_d = criterion.nce_loss(z_d_ref, z_d_pos, z_d_neg)

    l_rec = criterion.rec_loss(model_output["x_hat_lr"], lr_ref)
    l_sr = criterion.rec_loss(y_hat_hr, y_hr)
    l_kl_c = criterion.kl_loss(model_output["mu_c"], model_output["log_var_c"])
    l_kl_d = criterion.kl_loss(model_output["mu_d"], model_output["log_var_d"])
    l_kl = l_kl_c + l_kl_d
    l_prior = criterion.prior_loss(model_output["z_clean"])
    l_adv = criterion.adv_loss.generator_loss(d_fake_for_g)

    l_total = (
        criterion.lambda_rec * l_rec
        + criterion.lambda_sr * l_sr
        + criterion.lambda_kl * l_kl
        + criterion.lambda_nce * (l_nce_c + l_nce_d)
        + criterion.lambda_adv * l_adv
        + criterion.lambda_prior * l_prior
    )

    optimizer_G.zero_grad()
    l_total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer_G.step()

    loss_dict = {
        "l_rec": l_rec.item(),
        "l_sr": l_sr.item(),
        "l_kl": l_kl.item(),
        "l_nce_c": l_nce_c.item(),
        "l_nce_d": l_nce_d.item(),
        "l_adv": l_adv.item(),
        "l_prior": l_prior.item(),
        "l_total": l_total.item(),
        "d_loss": d_loss.item(),
    }

    return loss_dict


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(args.output_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_logs"))

    logger.info(f"Config: {config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    degradation_pipeline = DegradationPipeline(
        scale_factor=config["data"]["scale_factor"],
        blur_sig_x_range=tuple(config["degradation"]["blur_sig_x_range"]),
        blur_sig_y_range=tuple(config["degradation"]["blur_sig_y_range"]),
        blur_rot_range=tuple(config["degradation"]["blur_rot_range"]),
        noise_range=tuple(config["degradation"]["noise_range"]),
        jpeg_range=tuple(config["degradation"]["jpeg_range"]),
        second_order_prob=config["degradation"]["second_order_prob"],
    )

    dataloader = build_dataloader(
        hr_image_dirs=config["data"]["hr_image_dirs"],
        hr_patch_size=config["data"]["hr_patch_size"],
        scale_factor=config["data"]["scale_factor"],
        batch_size=config["train"]["batch_size"],
        num_workers=config["train"]["num_workers"],
        num_degradation_variants=config["train"]["num_degradation_variants"],
        degradation_pipeline=degradation_pipeline,
    )

    model = CISRModel(
        backbone_name=config["model"]["backbone_name"],
        latent_dim=config["model"]["latent_dim"],
        degradation_latent_dim=config["model"]["degradation_latent_dim"],
        decoder_num_features=config["model"]["decoder_num_features"],
        decoder_num_res_blocks=config["model"]["decoder_num_res_blocks"],
        scale_factor=config["data"]["scale_factor"],
        freeze_backbone=config["model"]["freeze_backbone"],
        num_heads=config["model"]["num_heads"],
        cafi_dropout=config["model"]["cafi_dropout"],
    ).to(device)

    criterion = CISRLoss(
        lambda_rec=config["loss"]["lambda_rec"],
        lambda_sr=config["loss"]["lambda_sr"],
        lambda_kl=config["loss"]["lambda_kl"],
        lambda_nce=config["loss"]["lambda_nce"],
        lambda_adv=config["loss"]["lambda_adv"],
        lambda_prior=config["loss"]["lambda_prior"],
        temperature=config["loss"]["temperature"],
    )

    gen_params = list(model.content_encoder.parameters()) + \
                 list(model.degradation_encoder.parameters()) + \
                 list(model.cafi.parameters()) + \
                 list(model.decoder.parameters()) + \
                 [model.z_clean]
    gen_params = [p for p in gen_params if p.requires_grad]

    optimizer_G = Adam(gen_params, lr=config["train"]["lr"],
                       betas=(config["train"]["beta1"], config["train"]["beta2"]))
    optimizer_D = Adam(model.discriminator.parameters(),
                       lr=config["train"]["lr"],
                       betas=(config["train"]["beta1"], config["train"]["beta2"]))

    total_iters = config["train"]["total_iters"]
    scheduler_G = CosineAnnealingLR(optimizer_G, T_max=total_iters, eta_min=1e-7)
    scheduler_D = CosineAnnealingLR(optimizer_D, T_max=total_iters, eta_min=1e-7)

    start_iter = 0
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_G.load_state_dict(checkpoint["optimizer_G_state_dict"])
        optimizer_D.load_state_dict(checkpoint["optimizer_D_state_dict"])
        start_iter = checkpoint.get("iteration", 0)

    logger.info("Starting training...")
    data_iter = iter(dataloader)

    for iteration in range(start_iter, total_iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        loss_dict = train_one_iteration(
            model, batch, criterion, optimizer_G, optimizer_D, device
        )

        scheduler_G.step()
        scheduler_D.step()

        if (iteration + 1) % config["train"]["log_interval"] == 0:
            lr_current = optimizer_G.param_groups[0]["lr"]
            log_str = f"Iter [{iteration+1}/{total_iters}] lr={lr_current:.2e}"
            for k, v in loss_dict.items():
                log_str += f" {k}={v:.4f}"
                writer.add_scalar(f"train/{k}", v, iteration)
            logger.info(log_str)

        if (iteration + 1) % config["train"]["save_interval"] == 0:
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_{iteration+1}.pth")
            torch.save({
                "iteration": iteration + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_G_state_dict": optimizer_G.state_dict(),
                "optimizer_D_state_dict": optimizer_D.state_dict(),
                "config": config,
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")

    final_path = os.path.join(args.output_dir, "checkpoint_final.pth")
    torch.save({
        "iteration": total_iters,
        "model_state_dict": model.state_dict(),
        "optimizer_G_state_dict": optimizer_G.state_dict(),
        "optimizer_D_state_dict": optimizer_D.state_dict(),
        "config": config,
    }, final_path)
    logger.info(f"Final checkpoint saved: {final_path}")

    writer.close()


if __name__ == "__main__":
    main()
