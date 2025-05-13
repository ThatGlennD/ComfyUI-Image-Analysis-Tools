import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import os

class ContrastAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["Global", "Local", "Hybrid"], {"default": "Hybrid"}),
                "comparison_method": (["Michelson", "RMS", "Weber"], {"default": "RMS"}),
                "block_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "visualize_contrast_map": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("contrast_score", "contrast_map")
    FUNCTION = "analyze"
    CATEGORY = "Image Analysis"

    def analyze(self, image, method, comparison_method, block_size, visualize_contrast_map):
        img_tensor = image[0]
        if img_tensor.ndim == 4:
            img_tensor = img_tensor[0]
        if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
            np_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        elif img_tensor.ndim == 3 and img_tensor.shape[2] in [1, 3]:
            np_img = img_tensor.cpu().numpy()
        else:
            raise ValueError(f"Unsupported image shape: {img_tensor.shape}")

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

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                plt.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
                plt.close(fig)
                img = cv2.imread(tmpfile.name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                os.unlink(tmpfile.name)
            tensor_img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            tensor_img = torch.zeros((1, 3, 64, 64), dtype=torch.float32)

        return float(score), tensor_img

NODE_CLASS_MAPPINGS = {
    "ContrastAnalysis": ContrastAnalysis
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ContrastAnalysis": "Contrast Analysis"
}