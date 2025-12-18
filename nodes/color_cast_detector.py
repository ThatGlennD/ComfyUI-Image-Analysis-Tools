import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import os
import io as py_io
from comfy_api.latest import io

class ColorCastDetector(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Color Cast Detector",
            display_name="Color Cast Detector",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("tolerance", default=0.05, min=0.01, max=0.5),
                io.Boolean.Input("visualize_color_bias", default=True),
                io.Enum.Input("visualization_mode", ["Channel Difference", "Neutrality Deviation"])
            ],
            outputs=[
                io.Float.Output("cast_score"),
                io.Image.Output("color_bias_map"),
                io.String.Output("interpretation")
            ]
        )
    @classmethod
    def execute(cls, image, tolerance, visualize_color_bias, visualization_mode) -> io.NodeOutput:
        try:
            img_tensor = image[0]
            # Standard ComfyUI is [H, W, 3]
            np_img = img_tensor.cpu().numpy()

            # If accidentally CHW (unlikely in standard pipeline but checking)
            if np_img.shape[0] == 3 and np_img.shape[2] > 3:
                 np_img = np_img.transpose(1, 2, 0)

            uint8_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
            mean_rgb = np.mean(uint8_img.reshape(-1, 3), axis=0)
            mean_norm = mean_rgb / np.sum(mean_rgb)

            ref = 1.0 / 3
            delta = mean_norm - ref
            cast_score = float(np.max(np.abs(delta)))

            dominant = np.argmax(delta)
            weakest = np.argmin(delta)

            channels = ['Red', 'Green', 'Blue']
            dominant_name = channels[dominant]
            weakest_name = channels[weakest]

            if cast_score < tolerance:
                interpretation = "No significant color cast"
            else:
                direction = f"{dominant_name} tint (Δ{dominant_name} = {delta[dominant]:.2f})"
                if (dominant_name == 'Red' and weakest_name == 'Green') or                    (dominant_name == 'Green' and weakest_name == 'Red'):
                    direction += " → Possible magenta/green cast"
                elif (dominant_name == 'Red' and weakest_name == 'Blue') or                      (dominant_name == 'Blue' and weakest_name == 'Red'):
                    direction += " → Possible cyan/red cast"
                elif (dominant_name == 'Green' and weakest_name == 'Blue') or                      (dominant_name == 'Blue' and weakest_name == 'Green'):
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
                        np.clip(diff_rb + 128, 0, 255)
                    ], axis=-1).astype(np.uint8)
                else:  # Neutrality Deviation
                    r, g, b = uint8_img[:, :, 0], uint8_img[:, :, 1], uint8_img[:, :, 2]
                    avg = ((r + g + b) / 3).astype(np.uint8)
                    deviation = np.abs(uint8_img - avg[:, :, np.newaxis])
                    dev_map = np.clip(deviation * 2, 0, 255).astype(np.uint8)
                    diff_map = dev_map

                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(diff_map)
                ax.axis("off")
                buf = py_io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches="tight", dpi=150)
                plt.close(fig)
                buf.seek(0)
                img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                vis_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

                map_tensor = torch.from_numpy(vis_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                map_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return io.NodeOutput(cast_score, map_tensor, interpretation)

        except Exception as e:
            print(f"[ColorCastDetector] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(0.0, fallback, "Error during processing")
