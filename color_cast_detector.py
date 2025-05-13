
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import os

class ColorCastDetector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "tolerance": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01}),
                "visualize_color_bias": ("BOOLEAN", {"default": True}),
                "visualization_mode": (["Channel Difference", "Neutrality Deviation"],)
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("cast_score", "color_bias_map", "interpretation")
    FUNCTION = "analyze"
    CATEGORY = "Image Analysis"

    def analyze(self, image, tolerance, visualize_color_bias, visualization_mode):
        try:
            img_tensor = image[0]
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
                np_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            else:
                np_img = img_tensor.cpu().numpy()

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
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    vis_img = cv2.imread(tmpfile.name)
                    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
                    os.unlink(tmpfile.name)

                map_tensor = torch.from_numpy(vis_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                map_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return cast_score, map_tensor, interpretation

        except Exception as e:
            print(f"[ColorCastDetector] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return 0.0, fallback, "Error during processing"

NODE_CLASS_MAPPINGS = {
    "Color Cast Detector": ColorCastDetector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Color Cast Detector": "Color Cast Detector"
}
