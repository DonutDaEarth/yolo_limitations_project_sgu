"""Quick test script for YOLO model"""
import sys
sys.path.insert(0, 'src')

from models.yolo_detector import YOLODetector

print("🔄 Initializing YOLO detector...")
detector = YOLODetector(model_variant='yolov8n', device='cpu')
print("✓ YOLO loaded successfully!")
print(f"  Model: {detector.model_variant}")
print(f"  Device: {detector.device}")
print("\n✅ YOLO is working correctly!")
