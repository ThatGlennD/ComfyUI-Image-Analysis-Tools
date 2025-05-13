
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import os

class NoiseEstimation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "block_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "visualize_noise_map": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE")
    RETURN_NAMES = ("noise_score", "noise_map")
    FUNCTION = "estimate"
    CATEGORY = "Image Analysis"

    def estimate(self, image, block_size, visualize_noise_map):
        try:
            img_tensor = image[0]
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
                np_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            elif img_tensor.ndim == 3 and img_tensor.shape[2] in [1, 3]:
                np_img = img_tensor.cpu().numpy()
            else:
                raise ValueError(f"Unsupported image shape: {img_tensor.shape}")

            uint8_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
            gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

            h, w = gray.shape
            h_blocks = h // block_size
            w_blocks = w // block_size

            heatmap = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            scores = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    var = np.var(block)
                    heatmap[i, j] = var
                    scores.append(var)

            global_score = float(np.mean(scores))

            if visualize_noise_map:
                vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                vis_up = cv2.resize(vis, (w, h), interpolation=cv2.INTER_NEAREST)

                # Generate visual with left-side colorbar
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(vis_up, cmap='jet', aspect='equal')
                ax.axis('off')

                cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label('Noise Strength (Variance)', fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.yaxis.set_label_position('left')
                cbar.ax.yaxis.set_ticks_position('left')

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    legend_img = cv2.imread(tmpfile.name)
                    legend_rgb = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
                    os.unlink(tmpfile.name)

                noise_map = legend_rgb.astype(np.float32) / 255.0
                noise_tensor = torch.from_numpy(noise_map).unsqueeze(0)

            else:
                noise_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return global_score, noise_tensor

        except Exception as e:
            print(f"[NoiseEstimation] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return 0.0, fallback

NODE_CLASS_MAPPINGS = {
    "Noise Estimation": NoiseEstimation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Noise Estimation": "Noise Estimation"
}
