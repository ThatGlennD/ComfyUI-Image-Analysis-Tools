
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import torch

class RGBHistogramRenderer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "view_histogram"
    OUTPUT_NODE = False
    CATEGORY = "image/analysis"

    def view_histogram(self, image):
        try:
            # Convert tensor to numpy HWC
            np_img = image[0].cpu().numpy().transpose(1, 2, 0)  # CHW → HWC
            red = np_img[:, :, 0]
            green = np_img[:, :, 1]
            blue = np_img[:, :, 2]

            # Plot histograms for R, G, B on same axes
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.hist(red.ravel(), bins=256, color='red', alpha=0.5, label='Red')
            ax.hist(green.ravel(), bins=256, color='green', alpha=0.5, label='Green')
            ax.hist(blue.ravel(), bins=256, color='blue', alpha=0.5, label='Blue')
            ax.set_title("RGB Histogram")
            ax.legend()
            fig.tight_layout()

            # Save plot to buffer as RGB PNG
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, transparent=False, facecolor='white')
            plt.close(fig)
            buf.seek(0)
            pil_image = Image.open(buf).convert("RGB")  # remove alpha if present

            # Convert to tensor in format (1, H, W, 3) float32 0-1
            img_np = np.array(pil_image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # (1, H, W, 3)

            return (img_tensor,)

        except Exception as e:
            print(f"[RGBHistogramInlineViewer] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (fallback,)

NODE_CLASS_MAPPINGS = {
    "RGB Histogram Renderer": RGBHistogramRenderer
}
