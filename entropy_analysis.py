
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import os

class EntropyAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "block_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "visualize_entropy_map": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("entropy_score", "entropy_map", "interpretation")
    FUNCTION = "analyze"
    CATEGORY = "Image Analysis"

    def compute_entropy(self, block):
        hist = cv2.calcHist([block], [0], None, [256], [0, 256])
        hist = hist.ravel()
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

    def interpret_entropy(self, score):
        if score < 2:
            return f"Very low entropy ({score:.2f} bits)"
        elif score < 4:
            return f"Low entropy ({score:.2f} bits)"
        elif score < 6:
            return f"Moderate entropy ({score:.2f} bits)"
        elif score < 7.5:
            return f"High entropy ({score:.2f} bits)"
        else:
            return f"Very high entropy ({score:.2f} bits)"

    def analyze(self, image, block_size, visualize_entropy_map):
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

            entropy_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            entropies = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    e = self.compute_entropy(block)
                    entropy_map[i, j] = e
                    entropies.append(e)

            global_entropy = float(np.mean(entropies))
            interpretation = self.interpret_entropy(global_entropy)

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

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    legend_img = cv2.imread(tmpfile.name)
                    legend_rgb = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
                    os.unlink(tmpfile.name)

                entropy_tensor = torch.from_numpy(legend_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                entropy_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return global_entropy, entropy_tensor, interpretation

        except Exception as e:
            print(f"[EntropyAnalysis] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return 0.0, fallback, "Error during processing"

NODE_CLASS_MAPPINGS = {
    "Entropy Analysis": EntropyAnalysis
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Entropy Analysis": "Entropy Analysis"
}
