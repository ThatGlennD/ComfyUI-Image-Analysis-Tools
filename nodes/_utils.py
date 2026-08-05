"""Shared helpers for the Image Analysis Toolkit nodes."""

from __future__ import annotations

import io as _io

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt


def to_gray(image: torch.Tensor) -> np.ndarray:
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] to a uint8 grayscale array.

    ComfyUI IMAGE tensors are always [B, H, W, C] float32 in [0, 1].
    Only the first frame of the batch is analyzed.
    """
    np_img = image[0].detach().cpu().numpy()
    np_img = np.clip(np_img, 0.0, 1.0)
    uint8_img = (np_img * 255.0).astype(np.uint8)
    if uint8_img.ndim == 2:
        return uint8_img
    if uint8_img.shape[2] == 1:
        return uint8_img[:, :, 0]
    return cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)


def to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a ComfyUI IMAGE tensor [B, H, W, C] to a uint8 RGB array (first frame)."""
    np_img = image[0].detach().cpu().numpy()
    np_img = np.clip(np_img, 0.0, 1.0)
    uint8_img = (np_img * 255.0).astype(np.uint8)
    if uint8_img.ndim == 2:
        return cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2RGB)
    if uint8_img.shape[2] == 4:
        uint8_img = uint8_img[:, :, :3]
    if uint8_img.shape[2] == 1:
        return cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2RGB)
    return uint8_img


def fig_to_tensor(fig) -> torch.Tensor:
    """Render a matplotlib Figure to a [1, H, W, 3] float32 tensor in [0, 1].

    Replaces the tempfile round-trip used across the original nodes.
    """
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    pil = Image.open(buf).convert("RGB")
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def empty_image() -> torch.Tensor:
    """A black 64x64 placeholder image tensor."""
    return torch.zeros((1, 64, 64, 3), dtype=torch.float32)


from PIL import Image  # noqa: E402