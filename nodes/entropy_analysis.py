import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from comfy_api.latest import io

from ._utils import fig_to_tensor, empty_image


class EntropyAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EntropyAnalysis",
            display_name="Entropy Analysis",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("block_size", default=32, min=8, max=128, step=8),
                io.Boolean.Input("visualize_entropy_map", default=True),
            ],
            outputs=[
                io.Float.Output("entropy_score"),
                io.Image.Output("entropy_map"),
                io.String.Output("interpretation"),
            ],
        )

    @staticmethod
    def compute_entropy(block):
        hist = cv2.calcHist([block], [0], None, [256], [0, 256])
        hist = hist.ravel()
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    @staticmethod
    def interpret_entropy(score):
        if score < 2:
            return f"Very low entropy ({score:.2f} bits)"
        elif score < 4:
            return f"Low entropy ({score:.2f} bits)"
        elif score < 6:
            return f"Moderate entropy ({score:.2f} bits)"
        elif score < 7.5:
            return f"High entropy ({score:.2f} bits)"
        return f"Very high entropy ({score:.2f} bits)"

    @classmethod
    def execute(cls, image, block_size, visualize_entropy_map):
        try:
            np_img = image[0].detach().cpu().numpy()
            np_img = np.clip(np_img, 0.0, 1.0)
            uint8_img = (np_img * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

            h, w = gray.shape
            h_blocks = h // block_size
            w_blocks = w // block_size

            entropy_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            entropies = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    e = cls.compute_entropy(block)
                    entropy_map[i, j] = e
                    entropies.append(e)

            global_entropy = float(np.mean(entropies)) if entropies else 0.0
            interpretation = cls.interpret_entropy(global_entropy)

            if visualize_entropy_map:
                vis_up = cv2.resize(entropy_map, (w, h), interpolation=cv2.INTER_NEAREST)
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(vis_up, cmap="inferno", vmin=0, vmax=8, aspect="equal")
                ax.axis("off")

                cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label("Entropy (bits)", fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.yaxis.set_label_position("left")
                cbar.ax.yaxis.set_ticks_position("left")

                entropy_tensor = fig_to_tensor(fig)
            else:
                entropy_tensor = empty_image()

            return global_entropy, entropy_tensor, interpretation

        except Exception as e:
            print(f"[EntropyAnalysis] Error: {e}")
            return 0.0, empty_image(), "Error during processing"