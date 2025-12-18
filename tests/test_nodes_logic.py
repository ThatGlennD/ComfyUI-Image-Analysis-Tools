"""
Standalone tests for Image Analysis Tools nodes.
Tests pure Python/OpenCV logic without requiring ComfyUI environment.

The challenge: Node classes inherit from io.ComfyNode, which doesn't exist
outside ComfyUI's runtime. We solve this by providing a real base class mock
BEFORE the import happens.
"""
import sys
import os

# Step 1: Create a module-like object for comfy_api.latest.io
# with a real ComfyNode base class (not MagicMock)
class MockIO:
    class ComfyNode:
        """Real base class that node classes can inherit from."""
        pass

    class NodeOutput:
        """Mock NodeOutput that just stores args."""
        def __init__(self, *args):
            self.values = args
        def __iter__(self):
            return iter(self.values)

    class Image:
        @staticmethod
        def Input(name, **kwargs):
            return ("IMAGE", {"name": name, **kwargs})
        @staticmethod
        def Output(name="image"):
            return ("IMAGE", {"name": name})

    class Float:
        @staticmethod
        def Input(name, **kwargs):
            return ("FLOAT", {"name": name, **kwargs})
        @staticmethod
        def Output(name="float"):
            return ("FLOAT", {"name": name})

    class Int:
        @staticmethod
        def Input(name, **kwargs):
            return ("INT", {"name": name, **kwargs})
        @staticmethod
        def Output(name="int"):
            return ("INT", {"name": name})

    class String:
        @staticmethod
        def Input(name, **kwargs):
            return ("STRING", {"name": name, **kwargs})
        @staticmethod
        def Output(name="string"):
            return ("STRING", {"name": name})

    class Boolean:
        @staticmethod
        def Input(name, **kwargs):
            return ("BOOLEAN", {"name": name, **kwargs})

    class Enum:
        @staticmethod
        def Input(name, options, **kwargs):
            return ("ENUM", {"name": name, "options": options, **kwargs})

    class Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


# Step 2: Install the mock into sys.modules BEFORE importing nodes
mock_io = MockIO()
sys.modules["comfy_api"] = type(sys)("comfy_api")
sys.modules["comfy_api.latest"] = type(sys)("comfy_api.latest")
sys.modules["comfy_api.latest"].io = mock_io
sys.modules["comfy_api.latest.io"] = mock_io

# Add project root to path
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

# Step 3: NOW import the nodes (they will inherit from our real MockIO.ComfyNode)
import numpy as np
import torch

from nodes.defocus_analysis import DefocusAnalysis
from nodes.blur_detection import BlurDetection
from nodes.sharpness_focus_score import SharpnessFocusScore
from nodes.entropy_analysis import EntropyAnalysis

import unittest


class TestStaticMethods(unittest.TestCase):
    """Verify helper methods are static and callable directly from the class."""

    def test_defocus_interpret(self):
        # Low score = sharp
        result = DefocusAnalysis.interpret(0.1)
        self.assertIn("Very sharp", result)

        # High score = blurry
        result = DefocusAnalysis.interpret(0.9)
        self.assertIn("Severe defocus", result)

    def test_blur_interpret(self):
        result = BlurDetection.interpret_blur(25)
        self.assertIn("Very blurry", result)

        result = BlurDetection.interpret_blur(400)
        self.assertIn("Very sharp", result)

    def test_sharpness_interpret(self):
        result = SharpnessFocusScore.interpret_score(50, "Laplacian")
        self.assertIn("Very blurry", result)

        result = SharpnessFocusScore.interpret_score(0.8, "Hybrid")
        self.assertIn("Very sharp", result)

    def test_entropy_interpret(self):
        result = EntropyAnalysis.interpret_entropy(1.0)
        self.assertIn("Very low entropy", result)

        result = EntropyAnalysis.interpret_entropy(7.0)
        self.assertIn("High entropy", result)


class TestFFTAnalysis(unittest.TestCase):
    """Test pure image processing logic."""

    def test_fft_analysis_returns_valid_outputs(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[20:40, 20:40] = 255  # Add a bright square

        score, heatmap, mask = DefocusAnalysis.fft_analysis(gray, "FFT Ratio (Sum)")

        self.assertIsInstance(score, (float, np.floating))
        self.assertEqual(heatmap.shape[2], 3)  # BGR image
        self.assertEqual(mask.shape[2], 3)

    def test_edge_width_analysis(self):
        gray = np.zeros((64, 64), dtype=np.uint8)
        gray[30:35, :] = 255  # Horizontal line

        score, edge_vis, mask_vis = DefocusAnalysis.edge_width_analysis(gray, "Sobel")

        self.assertIsInstance(score, (float, np.floating))


class TestEntropyComputation(unittest.TestCase):
    """Test entropy calculation."""

    def test_compute_entropy_uniform(self):
        # Uniform block = low entropy
        block = np.ones((32, 32), dtype=np.uint8) * 128
        entropy = EntropyAnalysis.compute_entropy(block)
        self.assertLess(entropy, 1.0)

    def test_compute_entropy_random(self):
        # Random block = high entropy
        np.random.seed(42)
        block = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        entropy = EntropyAnalysis.compute_entropy(block)
        self.assertGreater(entropy, 5.0)


class TestNodeExecution(unittest.TestCase):
    """Test that execute runs without instantiation errors."""

    def setUp(self):
        # Create a dummy image tensor (B, H, W, C)
        self.dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        self.dummy_image[0, 20:40, 20:40, :] = 1.0  # Bright square

    def test_blur_detection_execute(self):
        # This should NOT raise AttributeError about immutable instance
        result = BlurDetection.execute(self.dummy_image, 16, False)
        self.assertIsNotNone(result)

    def test_defocus_analysis_execute(self):
        result = DefocusAnalysis.execute(
            self.dummy_image, "FFT Ratio (Sum)", True, "Sobel"
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
