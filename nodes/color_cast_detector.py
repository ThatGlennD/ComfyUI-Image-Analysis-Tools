import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from comfy_api.latest import io

from ._utils import fig_to_tensor, empty_image


class ColorCastDetector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ColorCastDetector",
            display_name="Color Cast Detector",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("tolerance", default=0.05, min=0.01, max=0.5, step=0.01),
                io.Boolean.Input("visualize_color_bias", default=True),
                io.Combo.Input("visualization_mode", options=["Channel Difference", "Neutrality Deviation"], default="Channel Difference"),
            ],
            outputs=[
                io.Float.Output("cast_score"),
                io.Image.Output("color_bias_map"),
                io.String.Output("interpretation"),
            ],
        )

    @classmethod
    def execute(cls, image, tolerance, visualize_color_bias, visualization_mode):
        try:
            np_img = image[0].detach().cpu().numpy()
            np_img = np.clip(np_img, 0.0, 1.0)
            uint8_img = (np_img * 255.0).astype(np.uint8)
            mean_rgb = np.mean(uint8_img.reshape(-1, 3), axis=0)
            mean_norm = mean_rgb / np.sum(mean_rgb)

            ref = 1.0 / 3
            delta = mean_norm - ref
            cast_score = float(np.max(np.abs(delta)))

            dominant = int(np.argmax(delta))
            weakest = int(np.argmin(delta))

            channels = ['Red', 'Green', 'Blue']
            dominant_name = channels[dominant]
            weakest_name = channels[weakest]

            if cast_score < tolerance:
                interpretation = "No significant color cast"
            else:
                direction = f"{dominant_name} tint (Δ{dominant_name} = {delta[dominant]:.2f})"
                pair = {dominant_name, weakest_name}
                if pair == {"Red", "Green"}:
                    direction += " → Possible magenta/green cast"
                elif pair == {"Red", "Blue"}:
                    direction += " → Possible cyan/red cast"
                elif pair == {"Green", "Blue"}:
                    direction += " → Possible yellow/blue cast"
                interpretation = f"Color cast detected: {direction}"

            if visualize_color_bias:
                if visualization_mode == "Channel Difference":
                    diff_rg = uint8_img[:, :, 0].astype(np.int16) - uint8_img[:, :, 1].astype(np.int16)
                    diff_gb = uint8_img[:, :, 1].astype(np.int16) - uint8_img[:, :, 2].astype(np.int16)
                    diff_rb = uint8_img[:, :, 0].astype(np.int16) - uint8_img[:, :, 2].astype(np.int16)
                    diff_map = np.stack([
                        np.clip(diff_rg + 128, 0, 255),
                        np.clip(diff_gb + 128, 0, 255),
                        np.clip(diff_rb + 128, 0, 255),
                    ], axis=-1).astype(np.uint8)
                else:  # Neutrality Deviation
                    r, g, b = uint8_img[:, :, 0], uint8_img[:, :, 1], uint8_img[:, :, 2]
                    avg = ((r + g + b) / 3).astype(np.uint8)
                    deviation = np.abs(uint8_img.astype(np.int16) - avg[:, :, np.newaxis].astype(np.int16))
                    diff_map = np.clip(deviation * 2, 0, 255).astype(np.uint8)

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(diff_map)
                ax.axis("off")
                map_tensor = fig_to_tensor(fig)
            else:
                map_tensor = empty_image()

            return cast_score, map_tensor, interpretation

        except Exception as e:
            print(f"[ColorCastDetector] Error: {e}")
            return 0.0, empty_image(), "Error during processing"