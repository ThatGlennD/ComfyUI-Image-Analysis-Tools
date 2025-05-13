
import numpy as np
import cv2
import torch

class SharpnessFocusScore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["Laplacian", "Tenengrad", "Hybrid"], {"default": "Hybrid"}),
                "visualize_edges": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE", "STRING")
    RETURN_NAMES = ("sharpness_score", "edge_visualization", "interpretation")
    FUNCTION = "calculate"
    CATEGORY = "Image Analysis"

    def interpret_score(self, score, method):
        if method == "Laplacian":
            if score < 100:
                desc = "Very blurry"
            elif score < 300:
                desc = "Soft focus"
            elif score < 700:
                desc = "Moderately sharp"
            else:
                desc = "Very sharp"
            return f"{desc} (based on Laplacian — responds to fine texture and local contrast)"
        elif method == "Tenengrad":
            if score < 10000:
                desc = "Very blurry"
            elif score < 25000:
                desc = "Soft focus"
            elif score < 50000:
                desc = "Moderately sharp"
            else:
                desc = "Very sharp"
            return f"{desc} (based on Tenengrad — emphasizes strong edges and gradients)"
        elif method == "Hybrid":
            if score < 0.2:
                desc = "Very blurry"
            elif score < 0.4:
                desc = "Soft focus"
            elif score < 0.7:
                desc = "Moderately sharp"
            else:
                desc = "Very sharp"
            return f"{desc} (hybrid of Laplacian and Tenengrad)"
        else:
            return "Unknown method"

    def calculate(self, image, method, visualize_edges):
        try:
            img_tensor = image[0]
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
                np_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            elif img_tensor.ndim == 3 and img_tensor.shape[2] in [1, 3]:
                np_img = img_tensor.cpu().numpy()
            else:
                raise ValueError(f"Unhandled image shape: {img_tensor.shape}")

            uint8_img = (np_img * 255).astype(np.uint8)

            if uint8_img.ndim == 3 and uint8_img.shape[2] == 3:
                gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)
            elif uint8_img.ndim == 2:
                gray = uint8_img
            elif uint8_img.ndim == 3 and uint8_img.shape[2] == 1:
                gray = uint8_img[:, :, 0]
            else:
                raise ValueError("Invalid image shape for grayscale conversion.")

            if gray is None or gray.size == 0:
                raise ValueError("Grayscale image is empty.")

            # Always compute both
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            lap_score = lap.var()

            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            mag = np.sqrt(gx ** 2 + gy ** 2)
            ten_score = np.mean(mag ** 2)

            if method == "Laplacian":
                score = lap_score
                edges = np.abs(lap)
            elif method == "Tenengrad":
                score = ten_score
                edges = mag
            elif method == "Hybrid":
                lap_norm = np.clip(lap_score / 1500, 0, 1)
                ten_norm = np.clip(ten_score / 50000, 0, 1)
                score = (lap_norm + ten_norm) / 2
                edges = np.abs(lap) + mag  # optional composite
            else:
                raise ValueError(f"Unknown method: {method}")

            if visualize_edges:
                vis = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
                edge_img = np.array(vis_rgb).astype(np.float32) / 255.0
                edge_tensor = torch.from_numpy(edge_img).unsqueeze(0)
            else:
                edge_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            interpretation = self.interpret_score(score, method)
            return float(score), edge_tensor, interpretation

        except Exception as e:
            print(f"[SharpnessFocusScore] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return 0.0, fallback, "Error during processing"

NODE_CLASS_MAPPINGS = {
    "Sharpness / Focus Score": SharpnessFocusScore
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Sharpness / Focus Score": "Sharpness / Focus Score"
}
