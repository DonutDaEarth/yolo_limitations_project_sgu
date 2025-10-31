"""YOLO Limitations Project - Models Package"""

from .yolo_detector import YOLODetector

try:
    from .two_stage_detector import FasterRCNNDetector
    __all__ = ['YOLODetector', 'FasterRCNNDetector']
except ImportError:
    print("⚠️  Warning: Detectron2 not installed. Faster R-CNN will not be available.")
    print("   Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    __all__ = ['YOLODetector']
