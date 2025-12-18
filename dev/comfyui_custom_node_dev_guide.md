# The ComfyUI Custom Node Developer Guide

This guide covers the lifecycle of creating a custom node, from architecture and setup to backend logic, frontend customization, and publishing to the Registry.

## 1. Architecture Overview

ComfyUI operates on a **Client-Server model**:
*   **Server (Python):** Handles the heavy lifting—loading models, processing tensors (images/latents), and executing algorithms.
*   **Client (JavaScript):** Handles the UI, graph connections, and widgets in the browser.

**Node Communication:**
Data flows between nodes as Python objects (mostly `torch.Tensor`). The Client sends a workflow (JSON) to the Server. The Server executes the graph and sends images/status updates back to the Client via WebSockets.

---

## 2. Setting Up

### Prerequisites
*   Python (3.9+)
*   ComfyUI installed
*   `comfy-cli` installed

### Scaffolding a Project
The easiest way to start is using the CLI to generate the directory structure.

```bash
cd ComfyUI/custom_nodes
comfy node scaffold
# Follow the prompts to set project name, license, etc.
```

This creates a folder with `__init__.py` (for module loading), `pyproject.toml` (for registry metadata), and source directories.

---

## 3. Backend Development (Python)

The backend defines what your node *does*.

### The Node Class Structure (V1 Legacy vs V3 Modern)

While V1 is common, ComfyUI is migrating to a V3 schema for better typing and organization.

#### V1 Schema (Standard/Legacy)
A standard node is a Python class with specific attributes:

```python
class MyNode:
    # 1. Definition
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "int_val": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            },
            "optional": {
                "optional_mask": ("MASK",),
            },
            "hidden": {
                "node_id": "UNIQUE_ID", # Access internal ID
                "prompt": "PROMPT",     # Access full prompt object
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("output_image", "count") # Optional labels
    FUNCTION = "execute_logic"
    CATEGORY = "MyCategory"

    # 2. Execution Logic
    def execute_logic(self, image, int_val, optional_mask=None, node_id=None):
        # Processing logic here
        return (image, int_val) # Must return a tuple, even for single output!
```

#### V3 Schema (Modern)
Uses object-oriented definitions and strict typing.

```python
from comfy_api.latest import io

class MyNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MyNode",
            display_name="My Cool Node",
            category="MyCategory",
            inputs=[
                io.Image.Input("image"),
                io.Int.Input("int_val", default=0, min=0, max=10)
            ],
            outputs=[io.Image.Output()]
        )

    @classmethod
    def execute(cls, image, int_val) -> io.NodeOutput:
        # Logic
        return io.NodeOutput(image)
```

### Critical Data Types
You will mostly work with `torch.Tensor`.
*   **IMAGE:** Shape `[Batch, Height, Width, Channels]`. Channels are usually 3 (RGB). Values are floats `0.0` to `1.0`.
*   **MASK:** Shape `[Batch, Height, Width]` or `[Height, Width]`. Values `0.0` to `1.0`.
*   **LATENT:** A dictionary: `{'samples': Tensor[Batch, 4, H/8, W/8]}`.

> **Tip:** When processing masks, check dimensions. You often need to `unsqueeze` to match shapes for broadcasting.

### Advanced Execution Features
*   **Lazy Evaluation:** Avoid calculating upstream nodes if not needed (e.g., a switch node).
    *   Add `{"lazy": True}` to `INPUT_TYPES`.
    *   Implement `check_lazy_status` method.
*   **List Processing:** Handle batches or lists sequentially.
    *   Set `INPUT_IS_LIST = True` or `OUTPUT_IS_LIST = (True,)`.
*   **IS_CHANGED:** Control caching. Return a unique hash/value to force re-execution.
*   **Validation:** Implement `VALIDATE_INPUTS` to check static data before execution starts.

---

## 4. Frontend Development (JavaScript)

The frontend defines how your node *looks* and interacts in the browser.

### Setup
1. Create a `js` folder inside your node directory.
2. In your python `__init__.py`, export: `WEB_DIRECTORY = "./js"`.

### Registering an Extension
Create a `.js` file in your web directory:

```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "my.unique.extension.name",
    
    // Hook: Run on startup
    async setup() {
        // Add event listeners, etc.
    },

    // Hook: Modify nodes before they are registered
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeType.comfyClass === "MyNode") {
            // Modify specific node behavior
        }
    },
    
    // Hook: Add Context Menu items (New API)
    getCanvasMenuItems(canvas) {
        return [{ content: "My Action", callback: () => { alert("Hi"); } }];
    }
});
```

### Modern Frontend APIs
*   **Context Menus:** Do not monkey-patch prototypes. Use `getCanvasMenuItems` and `getNodeMenuItems`.
*   **Settings:** Register settings via `app.registerExtension({ settings: [...] })` to appear in the ComfyUI settings gear.
*   **Keybindings:** Register shortcuts via `keybindings: [...]`.
*   **Badges:** Add info to the "About" page via `aboutPageBadges`.

---

## 5. Documentation & Polish

### Node Documentation (Help)
Create a `docs` folder inside your `WEB_DIRECTORY`.
*   File: `WEB_DIRECTORY/docs/MyNodeName.md`
*   Content: Markdown description, images, and usage examples. This appears in the UI when a user clicks "Node Info".
*   **i18n:** Create subfolders like `docs/MyNodeName/zh.md` for localization.

### Workflow Templates
Help users get started by including examples.
*   Create an `example_workflows` folder in your root.
*   Add `.json` workflow files and optional `.jpg` thumbnails.

### Localization (i18n)
Support multiple languages by creating a `locales` folder.
*   Structure: `locales/en/nodeDefs.json`, `locales/zh/nodeDefs.json`.
*   Map node names and inputs/outputs to translated strings.

---

## 6. Publishing

### The Comfy Registry (Preferred)
The Registry powers the ComfyUI Manager and ensures semantic versioning and security.

1.  **Initialize Metadata:**
    ```bash
    comfy node init
    ```
    This creates `pyproject.toml`.
2.  **Configure `pyproject.toml`:**
    *   **[project]**: `name` (unique ID), `version` (semantic), `dependencies`.
    *   **[tool.comfy]**: `PublisherId`, `Icon`, `DisplayName`.
3.  **Publish:**
    *   Get a Publisher ID and API Key from [registry.comfy.org](https://registry.comfy.org).
    *   Run:
    ```bash
    comfy node publish
    ```
4.  **CI/CD:** Use the provided Github Action templates to auto-publish on release or push.

### Security Standards
*   **No `eval` or `exec`:** These are strictly prohibited.
*   **No subprocess pip installs:** Let the Manager handle dependencies via `requirements.txt`.
*   **No Obfuscation:** Code must be readable.

### Legacy Method (ComfyUI Manager)
If not using the Registry, ensure you have a git repository with a `requirements.txt` and optionally an `install.py`. Submit a PR to the custom-node-list in the Manager repo.

---

## 7. Troubleshooting Tips

*   **Binary Search:** If ComfyUI crashes, disable half your custom nodes, restart, and repeat to find the culprit.
*   **Frontend Issues:** Check the browser console (F12).
*   **Backend Issues:** Check the Python console/terminal.
*   **Validation:** If your node inputs turn red, check `VALIDATE_INPUTS` logic or type mismatches.
*   **Visual Debugging:** Use `PromptServer.instance.send_sync("message_id", data)` in Python and listen via `api.addEventListener` in JS to send debug data to the browser.
