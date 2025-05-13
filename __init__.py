from .rgb_histogram_renderer import RGBHistogramRenderer
from .sharpness_focus_score import SharpnessFocusScore
from .noise_estimation_basic import NoiseEstimation
from .contrast_analysis import ContrastAnalysis
from .entropy_analysis import EntropyAnalysis
from .blur_detection import BlurDetection
from .edge_density_analysis import EdgeDensityAnalysis
from .clipping_analysis import ClippingAnalysis
from .color_cast_detector import ColorCastDetector
from .color_harmony_analyzer import ColorHarmonyAnalyzer
from .color_temperature_estimator import ColorTemperatureEstimator

NODE_CLASS_MAPPINGS = {
    "RGB Histogram Renderer": RGBHistogramRenderer,
    "Sharpness / Focus Score": SharpnessFocusScore,
    "Noise Estimation": NoiseEstimation,
    "Contrast Analysis": ContrastAnalysis,
    "Entropy Analysis": EntropyAnalysis,
    "Blur Detection": BlurDetection,
    "Edge Density Analysis": EdgeDensityAnalysis,
    "Clipping Analysis": ClippingAnalysis,
    "Color Cast Detector": ColorCastDetector,
    "Color Harmony Analyzer": ColorHarmonyAnalyzer,
    "Color Temperature Estimator": ColorTemperatureEstimator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGB Histogram Renderer": "RGB Histogram Renderer",
    "Sharpness / Focus Score": "Sharpness/Focus Score",
    "Noise Estimation": "Noise Estimation",
    "Contrast Analysis": "Contrast Analysis",
    "Entropy Analysis": "Entropy Analysis",
    "Blur Detection": "Blur Detection",
    "Edge Density Analysis": "Edge Density Analysis",
    "Clipping Analysis": "Clipping Analysis",
    "Color Cast Detector": "Color Cast Detector",
    "Color Harmony Analyzer": "Color Harmony Analyzer",
    "Color Temperature Estimator": "Color Temperature Estimator"

}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]