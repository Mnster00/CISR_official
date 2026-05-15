import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from .degradation import DegradationPipeline


class CISRDataset(Dataset):
    """
    Dataset for CISR training.
    Loads HR images and generates LR counterparts on-the-fly
    using the Real-ESRGAN style degradation pipeline.
    Also constructs contrastive triplets for interventional training.
    """

    def __init__(
        self,
        hr_image_dirs,
        hr_patch_size=192,
        scale_factor=4,
        num_degradation_variants=8,
        degradation_pipeline=None,
    ):
        super().__init__()
        self.hr_patch_size = hr_patch_size
        self.scale_factor = scale_factor
        self.lr_patch_size = hr_patch_size // scale_factor
        self.num_degradation_variants = num_degradation_variants

        if degradation_pipeline is None:
            self.degradation = DegradationPipeline(scale_factor=scale_factor)
        else:
            self.degradation = degradation_pipeline

        self.image_paths = []
        if isinstance(hr_image_dirs, str):
            hr_image_dirs = [hr_image_dirs]
        for d in hr_image_dirs:
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                        self.image_paths.append(os.path.join(d, f))

    def __len__(self):
        return len(self.image_paths)

    def _load_and_crop_hr(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))

        _, h, w = img_tensor.shape
        if h < self.hr_patch_size or w < self.hr_patch_size:
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0),
                size=(max(h, self.hr_patch_size), max(w, self.hr_patch_size)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            _, h, w = img_tensor.shape

        top = random.randint(0, h - self.hr_patch_size)
        left = random.randint(0, w - self.hr_patch_size)
        hr_patch = img_tensor[:, top:top + self.hr_patch_size, left:left + self.hr_patch_size]

        if random.random() < 0.5:
            hr_patch = torch.flip(hr_patch, dims=[2])
        if random.random() < 0.5:
            hr_patch = torch.flip(hr_patch, dims=[1])

        return hr_patch

    def __getitem__(self, idx):
        hr_patch = self._load_and_crop_hr(idx)

        degradation_params_list = []
        lr_variants = []
        for _ in range(self.num_degradation_variants):
            params = self.degradation.sample_degradation_params()
            degradation_params_list.append(params)
            lr_variant = self.degradation.apply_specific_degradation(hr_patch, params)
            lr_h, lr_w = self.lr_patch_size, self.lr_patch_size
            lr_variant = torch.nn.functional.interpolate(
                lr_variant.unsqueeze(0),
                size=(lr_h, lr_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            lr_variants.append(lr_variant)

        ref_idx = 0
        pos_c_idx = 1 if self.num_degradation_variants > 1 else 0
        pos_d_idx = 0

        return {
            "hr": hr_patch,
            "lr_ref": lr_variants[ref_idx],
            "lr_pos_c": lr_variants[pos_c_idx],
            "lr_all": torch.stack(lr_variants),
            "degradation_params": degradation_params_list,
        }


def build_dataloader(
    hr_image_dirs,
    hr_patch_size=192,
    scale_factor=4,
    batch_size=8,
    num_workers=4,
    num_degradation_variants=8,
    degradation_pipeline=None,
):
    dataset = CISRDataset(
        hr_image_dirs=hr_image_dirs,
        hr_patch_size=hr_patch_size,
        scale_factor=scale_factor,
        num_degradation_variants=num_degradation_variants,
        degradation_pipeline=degradation_pipeline,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader
