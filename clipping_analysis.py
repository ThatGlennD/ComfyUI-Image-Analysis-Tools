
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import os

class ClippingAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["Highlight/Shadow Clipping", "Saturation Clipping"],),
                "threshold": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "visualize_clipping_map": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("clipping_score", "clipping_map", "interpretation")
    FUNCTION = "analyze"
    CATEGORY = "Image Analysis"

    def analyze(self, image, mode, threshold, visualize_clipping_map):
        try:
            img_tensor = image[0]
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
                np_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            else:
                np_img = img_tensor.cpu().numpy()

            uint8_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)

            h, w, _ = uint8_img.shape

            if mode == "Highlight/Shadow Clipping":
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
                shadows = gray <= threshold
                highlights = gray >= 255 - threshold
                mask = np.zeros_like(uint8_img)
                mask[shadows] = [0, 0, 255]      # blue for shadows
                mask[highlights] = [255, 0, 0]   # red for highlights
                total_clipped = np.count_nonzero(shadows | highlights)
                description = f"Clipped highlights/shadows: {100 * total_clipped / (h * w):.2f}%"

            else:  # Saturation Clipping
                hsv = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)
                s_channel = hsv[:, :, 1]  # saturation
                v_channel = hsv[:, :, 2]  # value
                saturation_mask = (s_channel >= 255 - threshold) & (v_channel >= 255 - threshold)
                mask = np.zeros_like(uint8_img)
                mask[saturation_mask] = [255, 0, 255]  # magenta for saturation clipping
                total_clipped = np.count_nonzero(saturation_mask)
                description = f"Saturation-clipped pixels: {100 * total_clipped / (h * w):.2f}%"

            score = total_clipped / (h * w)

            if visualize_clipping_map:
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(mask)
                ax.axis("off")

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    map_img = cv2.imread(tmpfile.name)
                    map_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
                    os.unlink(tmpfile.name)

                map_tensor = torch.from_numpy(map_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                map_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return float(score), map_tensor, description

        except Exception as e:
            print(f"[ClippingAnalysis] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return 0.0, fallback, "Error during processing"

NODE_CLASS_MAPPINGS = {
    "Clipping Analysis": ClippingAnalysis
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Clipping Analysis": "Clipping Analysis"
}
