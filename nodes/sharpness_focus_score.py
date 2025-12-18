import numpy as np
import cv2
import torch
from comfy_api.latest import io

class SharpnessFocusScore(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Sharpness / Focus Score",
            display_name="Sharpness / Focus Score",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("method", ["Laplacian", "Tenengrad", "Hybrid"], default="Hybrid"),
                io.Boolean.Input("visualize_edges", default=False),
            ],
            outputs=[
                io.Float.Output("sharpness_score"),
                io.Image.Output("edge_visualization"),
                io.String.Output("interpretation"),
            ]
        )

    @staticmethod
    def interpret_score(score, method):
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

    @classmethod
    def execute(cls, image, method, visualize_edges):
        try:
            img_tensor = image[0]
            # Standard ComfyUI is [H, W, 3]
            np_img = img_tensor.cpu().numpy()

            # If accidentally CHW (unlikely in standard pipeline but checking)
            if np_img.shape[0] == 3 and np_img.shape[2] > 3:
                 np_img = np_img.transpose(1, 2, 0)

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

            # Use static methods directly to avoid immutability issues with cls() instantiation

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

            interpretation = cls.interpret_score(score, method)
            return io.NodeOutput(float(score), edge_tensor, interpretation)

        except Exception as e:
            print(f"[SharpnessFocusScore] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(0.0, fallback, "Error during processing")
