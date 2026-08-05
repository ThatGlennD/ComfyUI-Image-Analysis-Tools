import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from comfy_api.latest import io

from ._utils import fig_to_tensor, empty_image


class ContrastAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ContrastAnalysis",
            display_name="Contrast Analysis",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("method", options=["Global", "Local", "Hybrid"], default="Hybrid"),
                io.Combo.Input("comparison_method", options=["Michelson", "RMS", "Weber"], default="RMS"),
                io.Int.Input("block_size", default=32, min=8, max=128, step=8),
                io.Boolean.Input("visualize_contrast_map", default=True),
            ],
            outputs=[
                io.Float.Output("contrast_score"),
                io.Image.Output("contrast_map"),
            ],
        )

    @classmethod
    def execute(cls, image, method, comparison_method, block_size, visualize_contrast_map):
        try:
            np_img = image[0].detach().cpu().numpy()
            np_img = np.clip(np_img, 0.0, 1.0)
            gray = cv2.cvtColor((np_img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            h, w = gray.shape
            blocks = []

            if method in ("Local", "Hybrid"):
                for y in range(0, h, block_size):
                    for x in range(0, w, block_size):
                        block = gray[y:y + block_size, x:x + block_size]
                        if block.size == 0:
                            continue
                        if comparison_method == "Michelson":
                            c = (block.max() - block.min()) / (block.max() + block.min() + 1e-6)
                        elif comparison_method == "RMS":
                            c = block.std()
                        elif comparison_method == "Weber":
                            c = (block.max() - block.mean()) / (block.mean() + 1e-6)
                        else:
                            c = 0.0
                        blocks.append(c)
                local_contrast = float(np.mean(blocks)) if blocks else 0.0
            else:
                local_contrast = 0.0

            if method in ("Global", "Hybrid"):
                if comparison_method == "Michelson":
                    global_contrast = (gray.max() - gray.min()) / (gray.max() + gray.min() + 1e-6)
                elif comparison_method == "RMS":
                    global_contrast = float(gray.std())
                elif comparison_method == "Weber":
                    global_contrast = (gray.max() - gray.mean()) / (gray.mean() + 1e-6)
                else:
                    global_contrast = 0.0
            else:
                global_contrast = 0.0

            if method == "Global":
                score = global_contrast
            elif method == "Local":
                score = local_contrast
            else:  # Hybrid
                score = (global_contrast + local_contrast) / 2

            if visualize_contrast_map and method != "Global":
                map_h = (h + block_size - 1) // block_size
                map_w = (w + block_size - 1) // block_size
                contrast_map = np.zeros((map_h, map_w), dtype=np.float32)
                for i, c in enumerate(blocks):
                    contrast_map[i // map_w, i % map_w] = c

                vis_up = cv2.resize(contrast_map, (w, h), interpolation=cv2.INTER_NEAREST)
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(vis_up, cmap="magma", aspect="equal")
                ax.axis("off")

                cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label("Contrast Strength", fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.yaxis.set_label_position("left")
                cbar.ax.yaxis.set_ticks_position("left")

                tensor_img = fig_to_tensor(fig)
            else:
                tensor_img = empty_image()

            return float(score), tensor_img

        except Exception as e:
            print(f"[ContrastAnalysis] Error: {e}")
            return 0.0, empty_image()