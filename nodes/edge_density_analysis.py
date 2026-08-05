import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from comfy_api.latest import io

from ._utils import fig_to_tensor, empty_image


class EdgeDensityAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EdgeDensityAnalysis",
            display_name="Edge Density Analysis",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("method", options=["Canny", "Sobel"], default="Canny"),
                io.Int.Input("block_size", default=32, min=8, max=128, step=8),
                io.Boolean.Input("visualize_edge_map", default=True),
            ],
            outputs=[
                io.Float.Output("edge_density_score"),
                io.Image.Output("edge_density_map"),
                io.String.Output("interpretation"),
                io.Image.Output("edge_preview"),
            ],
        )

    @classmethod
    def execute(cls, image, method, block_size, visualize_edge_map):
        try:
            np_img = image[0].detach().cpu().numpy()
            np_img = np.clip(np_img, 0.0, 1.0)
            uint8_img = (np_img * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

            if method == "Canny":
                edges = cv2.Canny(gray, 100, 200)
            else:
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges = cv2.magnitude(sobelx, sobely)
                edges = np.uint8(np.clip(edges / np.max(edges) * 255, 0, 255))

            h, w = edges.shape
            h_blocks = h // block_size
            w_blocks = w // block_size
            density_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            densities = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = edges[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    density = float(np.count_nonzero(block)) / block.size
                    density_map[i, j] = density
                    densities.append(density)

            global_density = float(np.mean(densities)) if densities else 0.0

            if global_density < 0.05:
                interp = f"Very smooth ({global_density:.2f})"
            elif global_density < 0.15:
                interp = f"Soft detail ({global_density:.2f})"
            elif global_density < 0.3:
                interp = f"Moderate detail ({global_density:.2f})"
            else:
                interp = f"Dense detail ({global_density:.2f})"

            # Edge overlay preview.
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

                map_tensor = fig_to_tensor(fig)
            else:
                map_tensor = empty_image()

            return global_density, map_tensor, interp, edge_tensor

        except Exception as e:
            print(f"[EdgeDensityAnalysis] Error: {e}")
            return 0.0, empty_image(), "Error during processing", empty_image()