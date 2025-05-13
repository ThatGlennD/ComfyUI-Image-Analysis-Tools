
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import tempfile
import os

class EdgeDensityAnalysis:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["Canny", "Sobel"], {"default": "Canny"}),
                "block_size": ("INT", {"default": 32, "min": 8, "max": 128, "step": 8}),
                "visualize_edge_map": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("FLOAT", "IMAGE", "STRING", "IMAGE")
    RETURN_NAMES = ("edge_density_score", "edge_density_map", "interpretation", "edge_preview")
    FUNCTION = "analyze"
    CATEGORY = "Image Analysis"

    def analyze(self, image, method, block_size, visualize_edge_map):
        try:
            img_tensor = image[0]
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            if img_tensor.ndim == 3 and img_tensor.shape[0] in [1, 3]:
                np_img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            else:
                np_img = img_tensor.cpu().numpy()

            uint8_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
            gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

            if method == "Canny":
                edges = cv2.Canny(gray, 100, 200)
            else:
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges = cv2.magnitude(sobelx, sobely)
                edges = np.uint8(np.clip(edges / np.max(edges) * 255, 0, 255))
                _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

            h, w = edges.shape
            h_blocks = h // block_size
            w_blocks = w // block_size
            density_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            densities = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = edges[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    edge_pixels = np.count_nonzero(block)
                    density = edge_pixels / (block_size * block_size)
                    density_map[i, j] = density
                    densities.append(density)

            global_density = float(np.mean(densities))

            if global_density < 0.05:
                interp = f"Very smooth ({global_density:.2f})"
            elif global_density < 0.15:
                interp = f"Soft detail ({global_density:.2f})"
            elif global_density < 0.3:
                interp = f"Moderate detail ({global_density:.2f})"
            else:
                interp = f"Dense detail ({global_density:.2f})"

            # Create preview with edges highlighted
            edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            edge_overlay = np.clip(uint8_img * 0.6 + edge_color * 0.4, 0, 255).astype(np.uint8)
            edge_tensor = torch.from_numpy(edge_overlay.astype(np.float32) / 255.0).unsqueeze(0)

            if visualize_edge_map:
                vis_up = cv2.resize(density_map, (w, h), interpolation=cv2.INTER_NEAREST)

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(vis_up, cmap="magma", vmin=0, vmax=1, aspect="equal")
                ax.axis("off")

                cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label("Edge Density", fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.yaxis.set_label_position("left")
                cbar.ax.yaxis.set_ticks_position("left")

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    plt.savefig(tmpfile.name, bbox_inches="tight", dpi=150)
                    plt.close(fig)
                    map_img = cv2.imread(tmpfile.name)
                    map_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
                    os.unlink(tmpfile.name)

                map_tensor = torch.from_numpy(map_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                map_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return global_density, map_tensor, interp, edge_tensor

        except Exception as e:
            print(f"[EdgeDensityAnalysis] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return 0.0, fallback, "Error during processing", fallback

NODE_CLASS_MAPPINGS = {
    "Edge Density Analysis": EdgeDensityAnalysis
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Edge Density Analysis": "Edge Density Analysis"
}
