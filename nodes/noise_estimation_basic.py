import numpy as np
import cv2
from matplotlib import pyplot as plt
from comfy_api.latest import io

from ._utils import fig_to_tensor, empty_image


class NoiseEstimation(io.ComfyNode):
    """Estimate visual noise via the smoothing-residual method.

    The image is blurred with a Gaussian filter; the difference between the
    original and the smoothed image is the residual noise. Local variance of
    that residual per block gives the noise heatmap; the mean is the score.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="NoiseEstimation",
            display_name="Noise Estimation",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("block_size", default=32, min=8, max=128, step=8),
                io.Boolean.Input("visualize_noise_map", default=True),
            ],
            outputs=[
                io.Float.Output("noise_score"),
                io.Image.Output("noise_map"),
            ],
        )

    @classmethod
    def execute(cls, image, block_size, visualize_noise_map):
        try:
            np_img = image[0].detach().cpu().numpy()
            np_img = np.clip(np_img, 0.0, 1.0)
            uint8_img = (np_img * 255.0).astype(np.uint8)
            gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Residual noise = original - Gaussian-smoothed.
            smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
            residual = gray - smoothed

            h, w = gray.shape
            h_blocks = h // block_size
            w_blocks = w // block_size

            heatmap = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            scores = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = residual[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                    var = float(np.var(block))
                    heatmap[i, j] = var
                    scores.append(var)

            global_score = float(np.mean(scores)) if scores else 0.0

            if visualize_noise_map:
                vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                vis_up = cv2.resize(vis, (w, h), interpolation=cv2.INTER_NEAREST)

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(vis_up, cmap="jet", aspect="equal")
                ax.axis("off")

                cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label("Noise Strength (Variance)", fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.yaxis.set_label_position("left")
                cbar.ax.yaxis.set_ticks_position("left")

                noise_tensor = fig_to_tensor(fig)
            else:
                noise_tensor = empty_image()

            return global_score, noise_tensor

        except Exception as e:
            print(f"[NoiseEstimation] Error: {e}")
            return 0.0, empty_image()