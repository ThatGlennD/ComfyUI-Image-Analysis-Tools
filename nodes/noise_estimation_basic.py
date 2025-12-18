import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import os
import io as py_io
from comfy_api.latest import io

class NoiseEstimation(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Noise Estimation",
            display_name="Noise Estimation",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("block_size", default=32, min=8, max=128),
                io.Boolean.Input("visualize_noise_map", default=True)
            ],
            outputs=[
                io.Float.Output("noise_score"),
                io.Image.Output("noise_map")
            ]
        )

    @classmethod
    def execute(cls, image, block_size, visualize_noise_map):
        try:
            img_tensor = image[0]
            # Standard ComfyUI is [H, W, 3]
            np_img = img_tensor.cpu().numpy()

            # If accidentally CHW (unlikely in standard pipeline but checking)
            if np_img.shape[0] == 3 and np_img.shape[2] > 3:
                 np_img = np_img.transpose(1, 2, 0)

            uint8_img = (np.clip(np_img, 0, 1) * 255).astype(np.uint8)
            gray = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2GRAY)

            h, w = gray.shape
            h_blocks = h // block_size
            w_blocks = w // block_size

            heatmap = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            scores = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    var = np.var(block)
                    heatmap[i, j] = var
                    scores.append(var)

            global_score = float(np.mean(scores))

            if visualize_noise_map:
                vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                vis_up = cv2.resize(vis, (w, h), interpolation=cv2.INTER_NEAREST)

                # Generate visual with left-side colorbar
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(vis_up, cmap='jet', aspect='equal')
                ax.axis('off')

                cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label('Noise Strength (Variance)', fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.yaxis.set_label_position('left')
                cbar.ax.yaxis.set_ticks_position('left')

                buf = py_io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig)
                buf.seek(0)
                img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                legend_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                legend_rgb = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)

                noise_map = legend_rgb.astype(np.float32) / 255.0
                noise_tensor = torch.from_numpy(noise_map).unsqueeze(0)

            else:
                noise_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return io.NodeOutput(global_score, noise_tensor)

        except Exception as e:
            print(f"[NoiseEstimation] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(0.0, fallback)
