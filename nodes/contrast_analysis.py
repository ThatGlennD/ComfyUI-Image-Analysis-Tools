import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import os
import io as py_io
from comfy_api.latest import io

class ContrastAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Contrast Analysis",
            display_name="Contrast Analysis",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("method", ["Global", "Local", "Hybrid"], default="Hybrid"),
                io.Enum.Input("comparison_method", ["Michelson", "RMS", "Weber"], default="RMS"),
                io.Int.Input("block_size", default=32, min=8, max=128),
                io.Boolean.Input("visualize_contrast_map", default=True)
            ],
            outputs=[
                io.Float.Output("contrast_score"),
                io.Image.Output("contrast_map")
            ]
        )

    @classmethod
    def execute(cls, image, method, comparison_method, block_size, visualize_contrast_map) -> io.NodeOutput:
        img_tensor = image[0]
        # Standard ComfyUI is [H, W, 3]
        np_img = img_tensor.cpu().numpy()

        # If accidentally CHW (unlikely in standard pipeline but checking)
        if np_img.shape[0] == 3 and np_img.shape[2] > 3:
             np_img = np_img.transpose(1, 2, 0)

        gray = cv2.cvtColor((np.clip(np_img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        blocks = []

        if method in ["Local", "Hybrid"]:
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
                    blocks.append(c)
            local_contrast = np.mean(blocks) if blocks else 0
        else:
            local_contrast = 0

        if method in ["Global", "Hybrid"]:
            if comparison_method == "Michelson":
                global_contrast = (gray.max() - gray.min()) / (gray.max() + gray.min() + 1e-6)
            elif comparison_method == "RMS":
                global_contrast = gray.std()
            elif comparison_method == "Weber":
                global_contrast = (gray.max() - gray.mean()) / (gray.mean() + 1e-6)
        else:
            global_contrast = 0

        if method == "Global":
            score = global_contrast
        elif method == "Local":
            score = local_contrast
        else:
            score = (global_contrast + local_contrast) / 2

        if visualize_contrast_map and method != "Global":
            map_h = (h + block_size - 1) // block_size
            map_w = (w + block_size - 1) // block_size
            contrast_map = np.zeros((map_h, map_w), dtype=np.float32)
            idx = 0
            for y in range(map_h):
                for x in range(map_w):
                    if idx < len(blocks):
                        contrast_map[y, x] = blocks[idx]
                        idx += 1

            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(contrast_map, cmap="inferno", aspect="equal")
            ax.axis("off")

            cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
            cbar = plt.colorbar(im, cax=cbar_ax)
            cbar.set_label("Local Contrast", fontsize=10)
            cbar.ax.tick_params(labelsize=8)
            cbar.ax.yaxis.set_label_position("left")
            cbar.ax.yaxis.set_ticks_position("left")

            buf = py_io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            buf.seek(0)
            img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor_img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            tensor_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return io.NodeOutput(float(score), tensor_img)
