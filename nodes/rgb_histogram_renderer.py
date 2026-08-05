import numpy as np
import torch
from matplotlib import pyplot as plt
from comfy_api.latest import io

from ._utils import fig_to_tensor, to_rgb_uint8


class RGBHistogramRenderer(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="RGBHistogramRenderer",
            display_name="RGB Histogram Renderer",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
            ],
            outputs=[
                io.Image.Output("image"),
            ],
        )

    @classmethod
    def execute(cls, image):
        try:
            np_img = to_rgb_uint8(image)
            red = np_img[:, :, 0]
            green = np_img[:, :, 1]
            blue = np_img[:, :, 2]

            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.hist(red.ravel(), bins=256, color='red', alpha=0.5, label='Red')
            ax.hist(green.ravel(), bins=256, color='green', alpha=0.5, label='Green')
            ax.hist(blue.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
            ax.set_title("RGB Histogram")
            ax.legend()
            fig.tight_layout()

            return (fig_to_tensor(fig),)

        except Exception as e:
            print(f"[RGBHistogramRenderer] Error: {e}")
            from ._utils import empty_image
            return (empty_image(),)