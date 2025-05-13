# ComfyUI-Image-Analysis-Tools

A suite of precision-focused analysis nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI), designed to evaluate image quality, structure, and color balance. These tools are built for artists, developers, and researchers who want to assess generated images beyond surface aesthetics.

## 📦 Features

- **Sharpness & Blur Detection** — Laplacian, Tenengrad, Hybrid focus scores, and local blur heatmaps.
- **Noise Estimation** — Measure residual noise and generate noise maps.
- **Contrast Analysis** — Local, global, or hybrid scoring using Michelson, RMS, or Weber methods.
- **Entropy Evaluation** — Quantify visual information and texture richness.
- **Clipping Detection** — Identify overexposure, underexposure, and saturation loss.
- **Color Cast & Temperature** — Detect unbalanced tones and compute Kelvin white point.
- **Color Harmony Analysis** — Detect complementary, triadic, and other harmony structures.
- **Edge Density** — Measure structural detail using Canny or Sobel methods.
- **RGB Histogram Renderer** — View red, green, and blue tone distributions.

## 📁 Installation

1. Clone or download this repository into your ComfyUI custom nodes folder:

```bash
git clone git@github.com:yourusername/ComfyUI-Image-Analysis-Tools.git

pip install -r requirements.txt

numpy
opencv-python
matplotlib
scikit-learn
Pillow
torch
