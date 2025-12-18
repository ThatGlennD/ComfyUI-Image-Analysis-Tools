# Developer Guidelines

This document provides instructions and guidelines for working on the Image Analysis Toolkit.

## Project Structure

- `nodes/`: Contains the implementation of the ComfyUI nodes. Each node is typically in its own file.
- `web/`: Contains frontend assets (JS/CSS) if applicable.
- `tests/`: Contains tests. (Note: If this directory is missing, please create it).
- `requirements.txt`: Python dependencies.

## Coding Standards

### ComfyUI V3 Node Schema

This project uses the ComfyUI V3 Node Schema.
- Nodes should inherit from `io.ComfyNode`.
- Inputs and outputs are defined using the `define_schema` class method, which returns an `io.Schema` object.
- The `io.Schema` contains `node_id`, `display_name`, `category`, `inputs`, and `outputs`.
- The execution logic is in the `execute` class method.
- Use `io.NodeOutput(...)` for return values.

#### V3 Anti-Patterns (CRITICAL)
- **Do NOT instantiate the node class** inside `execute` (e.g., `instance = cls()` or `instance = MyNode()`). V3 node instances are immutable/frozen after registration. Attempting to set attributes on them will raise an `AttributeError`.
- **Use `@staticmethod`** for all helper methods. Call them via `cls.method_name(...)` or `MyNode.method_name(...)` inside the `execute` classmethod.

### File Handling & Windows Compatibility
- **Avoid Temporary Files**: On Windows, `tempfile.NamedTemporaryFile` can cause "File in use" errors if a process (like `cv2.imread`) tries to access it while another handle is still open.
- **Use BytesIO for Visualizations**: When generating plots with Matplotlib, save the figure to a `BytesIO` buffer instead of a file on disk.

Example of safe visualization:
```python
import io as py_io
import cv2
import numpy as np

# ... inside execute ...
buf = py_io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
plt.close(fig)
buf.seek(0)
img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

### Image Data

- Images are passed as PyTorch tensors.
- Shape: `[Batch, Height, Width, Channels]` (B, H, W, C).
- Values are typically normalized float32 between 0.0 and 1.0.
- When processing with OpenCV, convert to NumPy, remove batch dimension (if batch=1), and scale to 0-255 uint8.

## Testing

- Tests are located in the `tests/` directory.
- Run tests using: `python -m pytest`
- **Standalone Tests**: Since ComfyUI and its venv may not be accessible, write tests that mock `comfy_api`. Test the pure Python/OpenCV logic by isolating it into static methods.
- **Mocking `comfy`**: The `comfy` package (specifically `comfy.io`) is not available in the development environment. It is mocked in `tests/mocks/comfy` (or should be mocked if writing new tests). Ensure your tests set up `sys.path` or mocks correctly to handle `import comfy.io`.

## Dependencies

- Ensure dependencies are listed in `requirements.txt`.
- Main libraries: `numpy`, `opencv-python`, `torch`, `matplotlib`, `scikit-learn`, `Pillow`.
