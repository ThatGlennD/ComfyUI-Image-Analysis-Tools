import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from comfy_api.latest import io

from ._utils import fig_to_tensor, to_rgb_uint8


class ColorTemperatureEstimator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ColorTemperatureEstimator",
            display_name="Color Temperature Estimator",
            category="Image Analysis/Color",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Int.Output("kelvin"),
                io.String.Output("temperature_label"),
                io.Image.Output("color_swatch"),
            ],
        )

    @staticmethod
    def _estimate_color_temperature(img_uint8):
        img_f = img_uint8.astype(np.float32) / 255.0
        avg = img_f.mean(axis=(0, 1)).flatten()[:3]
        r, g, b = avg

        X = 0.412453 * r + 0.357580 * g + 0.180423 * b
        Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        Z = 0.019334 * r + 0.119193 * g + 0.950227 * b
        denom = X + Y + Z + 1e-6
        x = X / denom
        y = Y / denom
        n = (x - 0.3320) / (0.1858 - y + 1e-6)
        cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33

        kelvin = int(round(cct))
        if kelvin < 3000:
            lab = "Warm"
        elif kelvin < 4500:
            lab = "Neutral"
        elif kelvin < 6500:
            lab = "Cool Daylight"
        else:
            lab = "Blueish / Overcast"

        return kelvin, lab, avg

    @classmethod
    def execute(cls, image):
        img_uint8 = to_rgb_uint8(image)
        kelvin, label, avg_rgb = cls._estimate_color_temperature(img_uint8)

        fig, ax = plt.subplots(figsize=(1.28, 0.64), dpi=100)
        ax.axis("off")
        swatch_arr = np.ones((64, 128, 3), dtype=np.float32) * avg_rgb.reshape(1, 1, 3)
        ax.imshow(swatch_arr)
        text_color = "black" if avg_rgb.sum() > 1.5 else "white"
        ax.text(0.02, 0.6, f"{kelvin}K", color=text_color, fontsize=12, transform=ax.transAxes)

        swatch_tensor = fig_to_tensor(fig)
        return kelvin, label, swatch_tensor