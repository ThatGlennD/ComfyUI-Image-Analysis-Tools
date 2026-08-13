from .nodes.rgb_histogram_renderer import RGBHistogramRenderer
from .nodes.sharpness_focus_score import SharpnessFocusScore
from .nodes.noise_estimation_basic import NoiseEstimation
from .nodes.contrast_analysis import ContrastAnalysis
from .nodes.entropy_analysis import EntropyAnalysis
from .nodes.blur_detection import BlurDetection
from .nodes.edge_density_analysis import EdgeDensityAnalysis
from .nodes.clipping_analysis import ClippingAnalysis
from .nodes.color_cast_detector import ColorCastDetector
from .nodes.color_harmony_analyzer import ColorHarmonyAnalyzer
from .nodes.color_temperature_estimator import ColorTemperatureEstimator
from .nodes.defocus_analysis import DefocusAnalysis

NODE_CLASS_MAPPINGS = {
    "RGBHistogramRenderer": RGBHistogramRenderer,
    "SharpnessFocusScore": SharpnessFocusScore,
    "NoiseEstimation": NoiseEstimation,
    "ContrastAnalysis": ContrastAnalysis,
    "EntropyAnalysis": EntropyAnalysis,
    "BlurDetection": BlurDetection,
    "EdgeDensityAnalysis": EdgeDensityAnalysis,
    "ClippingAnalysis": ClippingAnalysis,
    "ColorCastDetector": ColorCastDetector,
    "ColorHarmonyAnalyzer": ColorHarmonyAnalyzer,
    "ColorTemperatureEstimator": ColorTemperatureEstimator,
    "DefocusAnalysis": DefocusAnalysis,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBHistogramRenderer": "RGB Histogram Renderer",
    "SharpnessFocusScore": "Sharpness/Focus Score",
    "NoiseEstimation": "Noise Estimation",
    "ContrastAnalysis": "Contrast Analysis",
    "EntropyAnalysis": "Entropy Analysis",
    "BlurDetection": "Blur Detection",
    "EdgeDensityAnalysis": "Edge Density Analysis",
    "ClippingAnalysis": "Clipping Analysis",
    "ColorCastDetector": "Color Cast Detector",
    "ColorHarmonyAnalyzer": "Color Harmony Analyzer",
    "ColorTemperatureEstimator": "Color Temperature Estimator",
    "DefocusAnalysis": "Defocus Analysis",
}

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
