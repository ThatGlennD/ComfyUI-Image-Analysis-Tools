
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import os
from sklearn.cluster import KMeans

class ColorHarmonyAnalyzer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_clusters": ("INT", {"default": 3, "min": 2, "max": 8}),
                "visualize_harmony": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING", "IMAGE")
    RETURN_NAMES = ("harmony_score", "harmony_type", "hue_wheel_visual")
    FUNCTION = "analyze"
    CATEGORY = "Image Analysis"

    def hue_distance(self, h1, h2):
        return min(abs(h1 - h2), 180 - abs(h1 - h2))

    def match_harmony(self, hues):
        if not hues or len(hues) < 2:
            return "Insufficient hues", 0.0

        scores = {}
        diffs = [self.hue_distance(hues[i], hues[j]) for i in range(len(hues)) for j in range(i+1, len(hues))]

        if any(170 <= d <= 190 for d in diffs):
            scores["Complementary"] = 1.0
        if all(d < 30 for d in diffs):
            scores["Analogous"] = 1.0
        if any(110 <= d <= 130 for d in diffs) and len(hues) >= 3:
            scores["Triadic"] = 1.0

        if len(hues) >= 3:
            sorted_hues = np.sort(hues)
            for i in range(len(sorted_hues)):
                base = sorted_hues[i]
                others = sorted_hues[:i].tolist() + sorted_hues[i+1:].tolist()
                split1 = (base + 150) % 180
                split2 = (base + 210) % 180
                split_hits = sum(min(abs(o - s), 180 - abs(o - s)) < 20 for o in others for s in [split1, split2])
                if split_hits >= 2:
                    scores["Split-Complementary"] = 1.0
                    break

        if len(hues) >= 4:
            extended_hues = sorted(hues + [(hues[0] + 180) % 180])
            distances = np.diff(extended_hues)
            if len(distances) >= 4:
                std_dev = np.std(distances)
                if std_dev < 20:
                    scores["Square"] = 1.0
                elif all(40 <= d <= 70 for d in distances):
                    scores["Tetradic"] = 1.0

        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            return best[0], best[1]
        return "No clear harmony", 0.0

    def analyze(self, image, num_clusters, visualize_harmony):
        try:
            img_tensor = image[0]
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            np_img = img_tensor.cpu().numpy()

            if np_img.shape[0] in [1, 3]:
                np_img = np.transpose(np_img, (1, 2, 0))

            uint8_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
            if len(uint8_img.shape) == 2:
                uint8_img = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2RGB)
            elif len(uint8_img.shape) == 3 and uint8_img.shape[2] != 3:
                uint8_img = cv2.cvtColor(uint8_img, cv2.COLOR_GRAY2RGB)

            hsv_img = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2HSV)
            h = hsv_img[:, :, 0].reshape(-1, 1)
            kmeans = KMeans(n_clusters=num_clusters, n_init="auto").fit(h)

            if len(kmeans.cluster_centers_) == 0:
                return 0.0, "No dominant hues found", torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            dominant_hues = sorted([int(center[0]) for center in kmeans.cluster_centers_])
            if not dominant_hues:
                return 0.0, "No dominant hues found", torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            harmony_type, score = self.match_harmony(dominant_hues)

            if visualize_harmony:
                fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
                hue_angles = [2 * np.pi * h / 180 for h in dominant_hues]
                ax.set_theta_direction(-1)
                ax.set_theta_zero_location('N')
                ax.set_yticklabels([])
                ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
                ax.set_xticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°'])
                for hue in hue_angles:
                    ax.plot([hue], [1], marker='o', markersize=12, color=plt.cm.hsv(hue / (2 * np.pi)))
                ax.set_title(harmony_type, fontsize=10)

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    plt.savefig(tmpfile.name, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    vis = cv2.imread(tmpfile.name)
                    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                    os.unlink(tmpfile.name)
                vis_tensor = torch.from_numpy(vis_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                vis_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return float(score), harmony_type, vis_tensor

        except Exception as e:
            print(f"[ColorHarmonyAnalyzer] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return 0.0, "Error during processing", fallback

NODE_CLASS_MAPPINGS = {
    "Color Harmony Analyzer": ColorHarmonyAnalyzer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Color Harmony Analyzer": "Color Harmony Analyzer"
}
