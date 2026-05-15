import random
import math
import torch
import numpy as np
from scipy import ndimage
from scipy.interpolate import interp2d
import cv2


def _random_gaussian_kernel(sig_x_range, sig_y_range, rot_range):
    sig_x = random.uniform(*sig_x_range)
    sig_y = random.uniform(*sig_y_range)
    rot = random.uniform(*rot_range)
    kernel_size = int(np.ceil(np.maximum(sig_x, sig_y) * 3)) * 2 + 1
    kernel = _gaussian_kernel2d(kernel_size, sig_x, sig_y, rot)
    return kernel


def _gaussian_kernel2d(kernel_size, sig_x, sig_y, rotation):
    coords = np.arange(kernel_size) - kernel_size // 2
    x, y = np.meshgrid(coords, coords)
    theta = rotation * np.pi / 180.0
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-0.5 * (x_rot ** 2 / sig_x ** 2 + y_rot ** 2 / sig_y ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def _apply_blur(img, kernel):
    img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img
    if img_np.ndim == 3:
        img_np = img_np.transpose(1, 2, 0)
    result = ndimage.convolve(img_np, kernel[:, :, np.newaxis] if img_np.ndim == 3 else kernel)
    if isinstance(img, torch.Tensor):
        if result.ndim == 3:
            result = result.transpose(2, 0, 1)
        return torch.from_numpy(result.copy()).float()
    return result


def _add_noise(img, noise_level):
    if isinstance(img, torch.Tensor):
        noise = torch.randn_like(img) * noise_level
        return img + noise
    else:
        noise = np.random.randn(*img.shape).astype(np.float32) * noise_level
        return img + noise


def _jpeg_compress(img, quality):
    if isinstance(img, torch.Tensor):
        img_np = img.cpu().numpy()
        if img_np.ndim == 3:
            img_np = img_np.transpose(1, 2, 0)
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    else:
        img_np = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", img_np, encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if decoded.ndim == 3:
        decoded = decoded[:, :, ::-1]

    result = decoded.astype(np.float32) / 255.0
    if isinstance(img, torch.Tensor):
        if result.ndim == 3:
            result = result.transpose(2, 0, 1)
        return torch.from_numpy(result.copy()).float()
    return result


def _resize(img, scale_factor):
    if isinstance(img, torch.Tensor):
        img_np = img.cpu().numpy()
        is_tensor = True
        if img_np.ndim == 3:
            img_np = img_np.transpose(1, 2, 0)
    else:
        img_np = img.copy()
        is_tensor = False

    h, w = img_np.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if is_tensor:
        if resized.ndim == 3:
            resized = resized.transpose(2, 0, 1)
        return torch.from_numpy(resized.copy()).float()
    return resized


class DegradationPipeline:
    """
    Real-ESRGAN style high-order degradation pipeline.
    Simulates complex real-world corruptions through sequences of:
    - Anisotropic Gaussian blur
    - Poisson-Gaussian noise (Rician approximation)
    - JPEG compression with varying quality factors
    - Resize operations

    Supports both first-order and second-order degradation.
    """

    def __init__(
        self,
        scale_factor=4,
        blur_sig_x_range=(0.2, 3.0),
        blur_sig_y_range=(0.2, 3.0),
        blur_rot_range=(-180, 180),
        noise_range=(0, 25),
        jpeg_range=(30, 95),
        second_order_prob=0.5,
        resize_prob=(0.25, 0.25, 0.25, 0.25),
    ):
        self.scale_factor = scale_factor
        self.blur_sig_x_range = blur_sig_x_range
        self.blur_sig_y_range = blur_sig_y_range
        self.blur_rot_range = blur_rot_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        self.second_order_prob = second_order_prob
        self.resize_prob = resize_prob

    def _apply_first_order(self, img_hr):
        img = img_hr.clone() if isinstance(img_hr, torch.Tensor) else img_hr.copy()

        kernel = _random_gaussian_kernel(
            self.blur_sig_x_range, self.blur_sig_y_range, self.blur_rot_range
        )
        img = _apply_blur(img, kernel)

        noise_level = random.uniform(*self.noise_range) / 255.0
        if noise_level > 0:
            img = _add_noise(img, noise_level)

        jpeg_quality = random.randint(self.jpeg_range[0], self.jpeg_range[1])
        img = _jpeg_compress(img, jpeg_quality)

        img = _resize(img, 1.0 / self.scale_factor)

        return img

    def _apply_second_order(self, img_hr):
        img = img_hr.clone() if isinstance(img_hr, torch.Tensor) else img_hr.copy()

        resize_mode = random.choices(
            ["up", "down", "keep", "random"], weights=self.resize_prob
        )[0]
        if resize_mode == "up":
            sf = random.uniform(1.0, 1.5)
            img = _resize(img, sf)
        elif resize_mode == "down":
            sf = random.uniform(0.5, 1.0)
            img = _resize(img, sf)

        kernel = _random_gaussian_kernel(
            (0.1, 1.5), (0.1, 1.5), self.blur_rot_range
        )
        img = _apply_blur(img, kernel)

        noise_level = random.uniform(0, self.noise_range[1] * 0.5) / 255.0
        if noise_level > 0:
            img = _add_noise(img, noise_level)

        jpeg_quality = random.randint(self.jpeg_range[0], self.jpeg_range[1])
        img = _jpeg_compress(img, jpeg_quality)

        img = _resize(img, 1.0 / self.scale_factor)

        return img

    def __call__(self, img_hr):
        if random.random() < self.second_order_prob:
            return self._apply_second_order(img_hr)
        else:
            return self._apply_first_order(img_hr)

    def apply_specific_degradation(self, img_hr, degradation_params):
        img = img_hr.clone() if isinstance(img_hr, torch.Tensor) else img_hr.copy()

        if "blur_kernel" in degradation_params:
            img = _apply_blur(img, degradation_params["blur_kernel"])

        if "noise_level" in degradation_params:
            noise_level = degradation_params["noise_level"]
            if noise_level > 0:
                img = _add_noise(img, noise_level)

        if "jpeg_quality" in degradation_params:
            img = _jpeg_compress(img, degradation_params["jpeg_quality"])

        img = _resize(img, 1.0 / self.scale_factor)

        return img

    def sample_degradation_params(self):
        params = {}
        params["blur_kernel"] = _random_gaussian_kernel(
            self.blur_sig_x_range, self.blur_sig_y_range, self.blur_rot_range
        )
        params["noise_level"] = random.uniform(*self.noise_range) / 255.0
        params["jpeg_quality"] = random.randint(self.jpeg_range[0], self.jpeg_range[1])
        return params
