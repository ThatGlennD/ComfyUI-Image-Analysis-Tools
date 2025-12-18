import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import os
import io as py_io
from comfy_api.latest import io

class EntropyAnalysis(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Entropy Analysis",
            display_name="Entropy Analysis",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("block_size", default=32, min=8, max=128),
                io.Boolean.Input("visualize_entropy_map", default=True)
            ],
            outputs=[
                io.Float.Output("entropy_score"),
                io.Image.Output("entropy_map"),
                io.String.Output("interpretation")
            ]
        )

    @staticmethod
    def compute_entropy(block):
        hist = cv2.calcHist([block], [0], None, [256], [0, 256])
        hist = hist.ravel()
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log2(prob))
        return entropy

    @staticmethod
    def interpret_entropy(score):
        if score < 2:
            return f"Very low entropy ({score:.2f} bits)"
        elif score < 4:
            return f"Low entropy ({score:.2f} bits)"
        elif score < 6:
            return f"Moderate entropy ({score:.2f} bits)"
        elif score < 7.5:
            return f"High entropy ({score:.2f} bits)"
        else:
            return f"Very high entropy ({score:.2f} bits)"

    @classmethod
    def execute(cls, image, block_size, visualize_entropy_map):
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

            entropy_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            entropies = []

            # Use static methods directly to avoid immutability issues with cls() instantiation

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    e = cls.compute_entropy(block)
                    entropy_map[i, j] = e
                    entropies.append(e)

            global_entropy = float(np.mean(entropies))
            interpretation = cls.interpret_entropy(global_entropy)

            if visualize_entropy_map:
                vis_up = cv2.resize(entropy_map, (w, h), interpolation=cv2.INTER_NEAREST)

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(vis_up, cmap="inferno", vmin=0, vmax=8, aspect="equal")
                ax.axis("off")

                cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label("Entropy (bits)", fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.yaxis.set_label_position("left")
                cbar.ax.yaxis.set_ticks_position("left")

                buf = py_io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig)
                buf.seek(0)
                img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                legend_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                legend_rgb = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)

                entropy_tensor = torch.from_numpy(legend_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                entropy_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return io.NodeOutput(global_entropy, entropy_tensor, interpretation)

        except Exception as e:
            print(f"[EntropyAnalysis] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(0.0, fallback, "Error during processing")
