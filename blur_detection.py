
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import os

class BlurDetection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "block_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "visualize_blur_map": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("blur_score", "blur_map", "interpretation")
    FUNCTION = "analyze"
    CATEGORY = "Image Analysis"

    def interpret_blur(self, score):
        if score < 50:
            return f"Very blurry ({score:.1f})"
        elif score < 150:
            return f"Slightly blurry ({score:.1f})"
        elif score < 300:
            return f"Acceptably sharp ({score:.1f})"
        else:
            return f"Very sharp ({score:.1f})"

    def analyze(self, image, block_size, visualize_blur_map):
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

            blur_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            scores = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    lap = cv2.Laplacian(block, cv2.CV_64F)
                    var = np.var(lap)
                    blur_map[i, j] = var
                    scores.append(var)

            global_score = float(np.mean(scores))
            interpretation = self.interpret_blur(global_score)

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

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    blur_img = cv2.imread(tmpfile.name)
                    blur_rgb = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
                    os.unlink(tmpfile.name)

                blur_tensor = torch.from_numpy(blur_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                blur_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return global_score, blur_tensor, interpretation

        except Exception as e:
            print(f"[BlurDetection] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return 0.0, fallback, "Error during processing"

NODE_CLASS_MAPPINGS = {
    "Blur Detection": BlurDetection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Blur Detection": "Blur Detection"
}
