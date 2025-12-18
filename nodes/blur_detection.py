import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import os
import io as py_io
from comfy_api.latest import io

class BlurDetection(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Blur Detection",
            display_name="Blur Detection",
            category="Image Analysis",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("block_size", default=32, min=8, max=128),
                io.Boolean.Input("visualize_blur_map", default=True)
            ],
            outputs=[
                io.Float.Output("blur_score"),
                io.Image.Output("blur_map"),
                io.String.Output("interpretation")
            ]
        )

    @staticmethod
    def interpret_blur(score):
        if score < 50:
            return f"Very blurry ({score:.1f})"
        elif score < 150:
            return f"Slightly blurry ({score:.1f})"
        elif score < 300:
            return f"Acceptably sharp ({score:.1f})"
        else:
            return f"Very sharp ({score:.1f})"

    @classmethod
    def execute(cls, image, block_size, visualize_blur_map):
        try:
            # ComfyUI image batches are [B, H, W, C]
            img_tensor = image[0]

            # Check for standard [H, W, C] vs [C, H, W] ambiguity
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

            blur_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
            scores = []

            for i in range(h_blocks):
                for j in range(w_blocks):
                    block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    lap = cv2.Laplacian(block, cv2.CV_64F)
                    var = np.var(lap)
                    blur_map[i, j] = var
                    scores.append(var)

            global_score = float(np.mean(scores))

            # Use instance method? execute is classmethod in V3 guideline,
            # so we must create instance or make helper static.
            # However, the interpret_blur method doesn't use self state.
            interpretation = cls.interpret_blur(global_score)

            if visualize_blur_map:
                vis_up = cv2.resize(blur_map, (w, h), interpolation=cv2.INTER_NEAREST)

                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(vis_up, cmap="viridis", aspect="equal")
                ax.axis("off")

                cbar_ax = fig.add_axes([0.05, 0.2, 0.03, 0.6])
                cbar = plt.colorbar(im, cax=cbar_ax)
                cbar.set_label("Blur Strength (Laplacian Variance)", fontsize=10)
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.yaxis.set_label_position("left")
                cbar.ax.yaxis.set_ticks_position("left")

                buf = py_io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig)
                buf.seek(0)
                img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                blur_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                blur_rgb = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)

                blur_tensor = torch.from_numpy(blur_rgb.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                blur_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

            return io.NodeOutput(global_score, blur_tensor, interpretation)

        except Exception as e:
            print(f"[BlurDetection] Error: {e}")
            fallback = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return io.NodeOutput(0.0, fallback, "Error during processing")
