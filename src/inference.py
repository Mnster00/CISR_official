import os
import argparse
import yaml

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from src.models import CISRModel


def parse_args():
    parser = argparse.ArgumentParser(description="CISR Inference")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input LR image or directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config file")
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


def save_image(tensor, save_path):
    img_np = tensor.squeeze(0).cpu().numpy()
    img_np = np.clip(img_np.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(save_path)


def sliding_window_inference(model, x_lr, tile_size, tile_overlap, scale_factor):
    B, C, H, W = x_lr.shape
    hr_h, hr_w = H * scale_factor, W * scale_factor

    output = torch.zeros(B, 3, hr_h, hr_w, device=x_lr.device)
    count = torch.zeros(B, 1, hr_h, hr_w, device=x_lr.device)

    stride = tile_size - tile_overlap

    hann_window = _create_hann_window(tile_size * scale_factor, tile_overlap * scale_factor)

    for h_idx in range(0, H, stride):
        for w_idx in range(0, W, stride):
            h_end = min(h_idx + tile_size, H)
            w_end = min(w_idx + tile_size, W)
            h_start = h_end - tile_size if h_end - h_idx < tile_size else h_idx
            w_start = w_end - tile_size if w_end - w_idx < tile_size else w_idx

            tile_lr = x_lr[:, :, h_start:h_end, w_start:w_end]

            if tile_lr.shape[2] < tile_size or tile_lr.shape[3] < tile_size:
                padded = torch.zeros(B, C, tile_size, tile_size, device=x_lr.device)
                padded[:, :, :tile_lr.shape[2], :tile_lr.shape[3]] = tile_lr
                tile_lr = padded

            tile_hr = model.inference(tile_lr)

            if tile_lr.shape[2] < tile_size or tile_lr.shape[3] < tile_size:
                tile_hr = tile_hr[:, :, :tile_lr.shape[2] * scale_factor,
                                         :tile_lr.shape[3] * scale_factor]

            hr_h_start = h_start * scale_factor
            hr_w_start = w_start * scale_factor
            hr_h_end = hr_h_start + tile_hr.shape[2]
            hr_w_end = hr_w_start + tile_hr.shape[3]

            output[:, :, hr_h_start:hr_h_end, hr_w_start:hr_w_end] += \
                tile_hr * hann_window
            count[:, :, hr_h_start:hr_h_end, hr_w_start:hr_w_end] += hann_window

    output = output / count.clamp(min=1e-8)
    return output


def _create_hann_window(size, overlap):
    window_1d = torch.hann_window(size)
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)


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

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif os.path.isdir(args.input):
        image_paths = [
            os.path.join(args.input, f)
            for f in sorted(os.listdir(args.input))
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
    else:
        raise ValueError(f"Input path not found: {args.input}")

    for img_path in image_paths:
        print(f"Processing: {img_path}")
        x_lr = load_image(img_path).to(device)

        _, _, H, W = x_lr.shape
        if H > args.tile_size or W > args.tile_size:
            y_hr = sliding_window_inference(
                model, x_lr, args.tile_size, args.tile_overlap, args.scale_factor
            )
        else:
            y_hr = model.inference(x_lr)

        filename = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(args.output, f"{filename}_x{args.scale_factor}.png")
        save_image(y_hr, save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
