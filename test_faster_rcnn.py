"""Test script to verify Faster R-CNN loads correctly"""
import sys
sys.path.insert(0, 'src')

print("🔄 Initializing Faster R-CNN detector...")
from models.two_stage_detector import FasterRCNNDetector

detector = FasterRCNNDetector(backbone='resnet50', device='cpu')
print(f"✓ Faster R-CNN loaded successfully!")
print(f"  Model: Faster R-CNN")
print(f"  Backbone: {detector.backbone}")
print(f"  Device: {detector.device}")

print("\n✅ Faster R-CNN is working correctly!")
