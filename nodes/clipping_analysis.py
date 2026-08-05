import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from comfy_api.latest import io

from ._utils import fig_to_tensor, empty_image


class ClippingAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ClippingAnalysis",
            display_name="Clipping Analysis",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("mode", options=["Highlight/Shadow Clipping", "Saturation Clipping"], default="Highlight/Shadow Clipping"),
                io.Int.Input("threshold", default=5, min=1, max=50, step=1),
                io.Boolean.Input("visualize_clipping_map", default=True),
            ],
            outputs=[
                io.Float.Output("clipping_score"),
                io.Image.Output("clipping_map"),
                io.String.Output("interpretation"),
            ],
        )

    @classmethod
    def execute(cls, image, mode, threshold, visualize_clipping_map):
        try:
            np_img = image[0].detach().cpu().numpy()
            np_img = np.clip(np_img, 0.0, 1.0)
            uint8_img = (np_img * 255.0).astype(np.uint8)
            h, w, _ = uint8_img.shape

            if mode == "Highlight/Shadow Clipping":
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
                shadows = gray <= threshold
                highlights = gray >= 255 - threshold
                mask = np.zeros_like(uint8_img)
                mask[shadows] = [0, 0, 255]      # blue for shadows
                mask[highlights] = [255, 0, 0]   # red for highlights
                total_clipped = int(np.count_nonzero(shadows | highlights))
                description = f"Clipped highlights/shadows: {100 * total_clipped / (h * w):.2f}%"

            else:  # Saturation Clipping
                hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)
                s_channel = hsv[:, :, 1]
                v_channel = hsv[:, :, 2]
                saturation_mask = (s_channel >= 255 - threshold) & (v_channel >= 255 - threshold)
                mask = np.zeros_like(uint8_img)
                mask[saturation_mask] = [255, 0, 255]  # magenta for saturation clipping
                total_clipped = int(np.count_nonzero(saturation_mask))
                description = f"Saturation-clipped pixels: {100 * total_clipped / (h * w):.2f}%"

            score = total_clipped / (h * w)

            if visualize_clipping_map:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(mask)
                ax.axis("off")
                map_tensor = fig_to_tensor(fig)
            else:
                map_tensor = empty_image()

            return float(score), map_tensor, description

        except Exception as e:
            print(f"[ClippingAnalysis] Error: {e}")
            return 0.0, empty_image(), "Error during processing"