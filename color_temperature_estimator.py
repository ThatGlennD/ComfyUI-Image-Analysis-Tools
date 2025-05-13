import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import os

class ColorTemperatureEstimator:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("INT", "STRING", "IMAGE")
    RETURN_NAMES = ("kelvin", "temperature_label", "color_swatch")
    FUNCTION = "estimate"
    CATEGORY = "Image Analysis/Color"

    def estimate(self, image):
        img = image[0]

        # Normalize ComfyUI inputs to H×W×C float32 [0–1]
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
            if arr.ndim == 4:            # [B, C, H, W] → drop batch
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (1, 3):   # [C, H, W] → [H, W, C]
                arr = arr.transpose(1, 2, 0)
            elif arr.ndim == 3 and arr.shape[2] in (1, 3): # [H, W, C]
                pass
            else:
                raise ValueError(f"Unsupported tensor shape: {arr.shape}")
        elif isinstance(img, np.ndarray) and img.ndim == 3:
            arr = img
        else:
            raise TypeError(f"Unsupported image input: {type(img)}, shape={getattr(img,'shape',None)}")

        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        img_uint8 = (arr * 255).astype(np.uint8)

        # Compute color temperature
        kelvin, label, avg_rgb = self._estimate_color_temperature(img_uint8)

        # — Build a 64×128 swatch via Matplotlib and save/load just like your working nodes —
        # Prepare figure matching your other nodes' pattern
        fig, ax = plt.subplots(figsize=(1.28, 0.64), dpi=100)
        ax.axis("off")
        swatch_arr = np.ones((64, 128, 3), dtype=np.float32) * avg_rgb.reshape(1,1,3)
        ax.imshow(swatch_arr)
        # Overlay Kelvin text
        text_color = "black" if avg_rgb.sum() > 1.5 else "white"
        ax.text(0.02, 0.6, f"{kelvin}K", color=text_color, fontsize=12, transform=ax.transAxes)

        # Save to temporary file, read back, convert to torch tensor
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            tmp_path = tmpfile.name
        img_bgr = cv2.imread(tmp_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        os.unlink(tmp_path)
        swatch_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        return kelvin, label, swatch_tensor

    def _estimate_color_temperature(self, img_uint8):
        img_f = img_uint8.astype(np.float32) / 255.0
        avg = img_f.mean(axis=(0,1)).flatten()[:3]
        r, g, b = avg

        X = 0.412453*r + 0.357580*g + 0.180423*b
        Y = 0.212671*r + 0.715160*g + 0.072169*b
        Z = 0.019334*r + 0.119193*g + 0.950227*b
        denom = X + Y + Z + 1e-6
        x = X/denom; y = Y/denom
        n = (x - 0.3320)/(0.1858 - y + 1e-6)
        CCT = 449*n**3 + 3525*n**2 + 6823.3*n + 5520.33

        kelvin = int(round(CCT))
        if kelvin < 3000:
            lab = "Warm"
        elif kelvin < 4500:
            lab = "Neutral"
        elif kelvin < 6500:
            lab = "Cool Daylight"
        else:
            lab = "Blueish / Overcast"

        return kelvin, lab, avg

NODE_CLASS_MAPPINGS = {
    "ColorTemperatureEstimator": ColorTemperatureEstimator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorTemperatureEstimator": "Color Temperature Estimator"
}