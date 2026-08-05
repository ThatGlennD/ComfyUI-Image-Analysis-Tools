import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from comfy_api.latest import io

from ._utils import fig_to_tensor, empty_image


class BlurDetection(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BlurDetection",
            display_name="Blur Detection",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("block_size", default=32, min=8, max=128, step=8),
                io.Boolean.Input("visualize_blur_map", default=True),
            ],
            outputs=[
                io.Float.Output("blur_score"),
                io.Image.Output("blur_map"),
                io.String.Output("interpretation"),
            ],
        )

    @staticmethod
    def interpret_blur(score):
        if score < 50:
            return f"Very blurry ({score:.1f})"
        elif score < 150:
            return f"Slightly blurry ({score:.1f})"
        elif score < 300:
            return f"Acceptably sharp ({score:.1f})"
        return f"Very sharp ({score:.1f})"

    @classmethod
    def execute(cls, image, block_size, visualize_blur_map):
        try:
            np_img = image[0].detach().cpu().numpy()
            np_img = np.clip(np_img, 0.0, 1.0)
            uint8_img = (np_img * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

            h, w = gray.shape
            h_blocks = h // block_size
            w_blocks = w // block_size

            blur_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            scores = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    lap = cv2.Laplacian(block, cv2.CV_64F)
                    var = float(np.var(lap))
                    blur_map[i, j] = var
                    scores.append(var)

            global_score = float(np.mean(scores)) if scores else 0.0
            interpretation = cls.interpret_blur(global_score)

            if visualize_blur_map:
                vis_up = cv2.resize(blur_map, (w, h), interpolation=cv2.INTER_NEAREST)
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(vis_up, cmap="viridis", aspect="equal")
                ax.axis("off")

                cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label("Blur Strength (Laplacian Variance)", fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.yaxis.set_label_position("left")
                cbar.ax.yaxis.set_ticks_position("left")

                blur_tensor = fig_to_tensor(fig)
            else:
                blur_tensor = empty_image()

            return global_score, blur_tensor, interpretation

        except Exception as e:
            print(f"[BlurDetection] Error: {e}")
            return 0.0, empty_image(), "Error during processing"