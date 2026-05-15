import os
import argparse
import yaml

import torch
import numpy as np
from PIL import Image

from src.models import CISRModel


def parse_args():
    parser = argparse.ArgumentParser(description="CISR Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
    parser.add_argument("--lr_dir", type=str, required=True,
                        help="Directory containing LR test images")
    parser.add_argument("--hr_dir", type=str, default=None,
                        help="Directory containing HR ground truth images")
    parser.add_argument("--output_dir", type=str, default="results/eval",
                        help="Output directory for results")
    parser.add_argument("--scale_factor", type=int, default=4,
                        help="Super-resolution scale factor")
    parser.add_argument("--tile_size", type=int, default=128,
                        help="Tile size for sliding window inference")
    parser.add_argument("--tile_overlap", type=int, default=36,
                        help="Overlap between tiles")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)
    return img_tensor


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def compute_ssim(img1, img2, window_size=11):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    from scipy.ndimage import uniform_filter

    mu1 = uniform_filter(img1, size=window_size)
    mu2 = uniform_filter(img2, size=window_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 ** 2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=window_size) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def sliding_window_inference(model, x_lr, tile_size, tile_overlap, scale_factor):
    from src.inference import sliding_window_inference as swi
    return swi(model, x_lr, tile_size, tile_overlap, scale_factor)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = CISRModel(
        backbone_name=config["model"]["backbone_name"],
        latent_dim=config["model"]["latent_dim"],
        degradation_latent_dim=config["model"]["degradation_latent_dim"],
        decoder_num_features=config["model"]["decoder_num_features"],
        decoder_num_res_blocks=config["model"]["decoder_num_res_blocks"],
        scale_factor=args.scale_factor,
        freeze_backbone=config["model"]["freeze_backbone"],
        num_heads=config["model"]["num_heads"],
        cafi_dropout=config["model"]["cafi_dropout"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    lr_images = sorted([
        f for f in os.listdir(args.lr_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ])

    psnr_list = []
    ssim_list = []

    for img_name in lr_images:
        lr_path = os.path.join(args.lr_dir, img_name)
        x_lr = load_image(lr_path).to(device)

        _, _, H, W = x_lr.shape
        if H > args.tile_size or W > args.tile_size:
            y_sr = sliding_window_inference(
                model, x_lr, args.tile_size, args.tile_overlap, args.scale_factor
            )
        else:
            y_sr = model.inference(x_lr)

        sr_np = y_sr.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        sr_np = np.clip(sr_np * 255.0, 0, 255).astype(np.uint8)

        sr_img = Image.fromarray(sr_np)
        save_path = os.path.join(args.output_dir, img_name)
        sr_img.save(save_path)

        if args.hr_dir is not None:
            hr_path = os.path.join(args.hr_dir, img_name)
            if os.path.exists(hr_path):
                hr_img = np.array(Image.open(hr_path).convert("RGB")).astype(np.float64)

                h_min = min(sr_np.shape[0], hr_img.shape[0])
                w_min = min(sr_np.shape[1], hr_img.shape[1])
                sr_crop = sr_np[:h_min, :w_min, :].astype(np.float64) / 255.0
                hr_crop = hr_img[:h_min, :w_min, :].astype(np.float64) / 255.0

                psnr_val = compute_psnr(sr_crop, hr_crop)
                ssim_val = compute_ssim(sr_crop * 255, hr_crop * 255)

                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

                print(f"{img_name}: PSNR={psnr_val:.4f}, SSIM={ssim_val:.4f}")

    if psnr_list:
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        print(f"\nAverage PSNR: {avg_psnr:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")

        results_path = os.path.join(args.output_dir, "evaluation_results.txt")
        with open(results_path, "w") as f:
            f.write(f"Dataset: {args.lr_dir}\n")
            f.write(f"Scale Factor: {args.scale_factor}\n")
            f.write(f"Number of Images: {len(psnr_list)}\n")
            f.write(f"Average PSNR: {avg_psnr:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            for i, img_name in enumerate(lr_images):
                if i < len(psnr_list):
                    f.write(f"  {img_name}: PSNR={psnr_list[i]:.4f}, SSIM={ssim_list[i]:.4f}\n")


if __name__ == "__main__":
    main()
