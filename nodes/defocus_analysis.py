import numpy as np
import cv2
import torch
from comfy_api.latest import io

class DefocusAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Defocus Analysis",
            display_name="Defocus Analysis",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Enum.Input("method", ["FFT Ratio (Sum)", "FFT Ratio (Mean)", "Hybrid (Mean+Sum)", "Edge Width"]),
                io.Boolean.Input("normalize", default=True),
                io.Enum.Input("edge_detector", ["Sobel", "Canny"], default="Sobel", optional=True),
            ],
            outputs=[
                io.Float.Output("defocus_score"),
                io.String.Output("interpretation"),
                io.Image.Output("fft_heatmap"),
                io.Image.Output("high_freq_mask"),
            ]
        )

    @classmethod
    def execute(cls, image, method, normalize=True, edge_detector="Sobel"):
        image_np = image[0].cpu().numpy()

        # If accidentally CHW
        if image_np.ndim == 3 and image_np.shape[0] == 3 and image_np.shape[2] > 3:
             image_np = image_np.transpose(1, 2, 0)

        image_np = (image_np * 255).astype(np.uint8)
        if image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        score = 0.0
        fft_vis = np.zeros((64, 64, 3), dtype=np.uint8)
        mask_vis = np.zeros((64, 64, 3), dtype=np.uint8)

        # Use static methods directly to avoid immutability issues with cls() instantiation

        if "FFT" in method or method.startswith("Hybrid"):
            score, fft_vis, mask_vis = cls.fft_analysis(gray, method)
        elif method == "Edge Width":
            score, fft_vis, mask_vis = cls.edge_width_analysis(gray, edge_detector)

        if normalize:
            score = max(0.0, min(score, 1.0))

        interpretation = cls.interpret(score)

        fft_tensor = torch.from_numpy(cv2.cvtColor(fft_vis, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_vis.astype(np.float32) / 255.0).unsqueeze(0)

        return io.NodeOutput(score, interpretation, fft_tensor, mask_tensor)

    @staticmethod
    def fft_analysis(gray, method):
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        log_mag = np.log1p(magnitude)
        norm = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

        cx, cy = magnitude.shape[1] // 2, magnitude.shape[0] // 2
        y, x = np.ogrid[:gray.shape[0], :gray.shape[1]]
        radius = min(cx, cy) // 4
        mask = (x - cx)**2 + (y - cy)**2 > radius**2

        hf_mag = magnitude * mask
        masked_norm = np.log1p(hf_mag)
        masked_vis = cv2.normalize(masked_norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mask_img = cv2.applyColorMap(masked_vis, cv2.COLORMAP_TURBO)

        sum_val = np.sum(magnitude[mask])
        mean_val = np.mean(magnitude[mask])
        total_val = np.sum(magnitude)

        fft_score_sum = 1.0 - (sum_val / total_val)
        fft_score_mean = 1.0 - (mean_val / magnitude.mean())

        if method == "FFT Ratio (Sum)":
            score = fft_score_sum
        elif method == "FFT Ratio (Mean)":
            score = fft_score_mean
        else:  # Hybrid
            score = 0.5 * fft_score_sum + 0.5 * fft_score_mean

        return score, heatmap, mask_img

    @staticmethod
    def edge_width_analysis(gray, detector):
        if detector not in ["Sobel", "Canny"]:
            detector = "Sobel"

        if detector == "Sobel":
            edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0) + cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        else:
            edges = cv2.Canny(gray, 100, 200)

        abs_edges = np.abs(edges)
        edge_vis = cv2.normalize(abs_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edge_vis_color = cv2.applyColorMap(edge_vis, cv2.COLORMAP_TURBO)

        thresholded = (abs_edges > np.mean(abs_edges)).astype(np.uint8)
        dilated = cv2.dilate(thresholded, np.ones((3, 3)))
        eroded = cv2.erode(thresholded, np.ones((3, 3)))
        thickness_map = dilated - eroded

        mask_vis = cv2.merge([(thickness_map * 255).astype(np.uint8)]*3)

        score = np.mean(thickness_map)

        return score, edge_vis_color, mask_vis

    @staticmethod
    def laplacian_blur_score(gray):
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return 1.0 - min(lap_var / 1000.0, 1.0)

    @staticmethod
    def interpret(score):
        if score < 0.2:
            return f"Very sharp — no defocus ({score:.2f})"
        elif score < 0.4:
            return f"Slight defocus — likely usable ({score:.2f})"
        elif score < 0.6:
            return f"Moderate defocus detected ({score:.2f})"
        elif score < 0.8:
            return f"Significant blur — check focus ({score:.2f})"
        else:
            return f"Severe defocus — image degraded ({score:.2f})"
