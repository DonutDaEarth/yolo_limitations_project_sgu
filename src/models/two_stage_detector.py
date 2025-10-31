"""
Two-Stage Detector Wrapper (Faster R-CNN)
Provides a unified interface for Faster R-CNN using Detectron2
"""

import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from typing import List, Dict
import time


class FasterRCNNDetector:
    """Wrapper class for Faster R-CNN with consistent interface"""
    
    def __init__(self, backbone='resnet50', weights=None, conf_threshold=0.25, 
                 iou_threshold=0.5, device='cuda'):
        """
        Initialize Faster R-CNN detector
        
        Args:
            backbone: Backbone network (resnet50, resnet101)
            weights: Path to config or weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.backbone = backbone
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Setup config
        cfg = get_cfg()
        
        if weights is None:
            # Use default pre-trained model
            if backbone == 'resnet50':
                config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
            elif backbone == 'resnet101':
                config_file = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        else:
            cfg.merge_from_file(weights)
        
        # Set confidence threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = iou_threshold
        
        # Set device
        cfg.MODEL.DEVICE = device
        
        print(f"Loading Faster R-CNN with {backbone} backbone")
        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg
        
        print(f"Faster R-CNN loaded successfully on {device}")
    
    def predict(self, image: np.ndarray, conf_threshold=None, iou_threshold=None) -> Dict:
        """
        Run inference on a single image
        
        Args:
            image: Input image (numpy array in BGR format)
            conf_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            
        Returns:
            Dictionary containing:
                - boxes: Bounding boxes (N, 4) in [x1, y1, x2, y2] format
                - scores: Confidence scores (N,)
                - classes: Class IDs (N,)
                - inference_time: Time taken for inference in seconds
        """
        # Update thresholds if provided
        if conf_threshold is not None:
            self.predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_threshold
        if iou_threshold is not None:
            self.predictor.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = iou_threshold
        
        # Measure inference time
        start_time = time.time()
        
        outputs = self.predictor(image)
        
        inference_time = time.time() - start_time
        
        # Extract predictions
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()  # [x1, y1, x2, y2]
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        # Reset thresholds
        if conf_threshold is not None:
            self.predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf_threshold
        if iou_threshold is not None:
            self.predictor.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.iou_threshold
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'inference_time': inference_time,
            'num_detections': len(boxes)
        }
    
    def predict_batch(self, images: List[np.ndarray], conf_threshold=None, 
                      iou_threshold=None) -> List[Dict]:
        """
        Run inference on a batch of images
        
        Args:
            images: List of input images
            conf_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        for image in images:
            pred = self.predict(image, conf_threshold, iou_threshold)
            predictions.append(pred)
        
        return predictions
    
    def benchmark(self, image: np.ndarray, num_iterations=100, warmup=10) -> Dict:
        """
        Benchmark inference speed
        
        Args:
            image: Input image
            num_iterations: Number of iterations for benchmarking
            warmup: Number of warmup iterations
            
        Returns:
            Dictionary with timing statistics
        """
        # Warmup
        for _ in range(warmup):
            self.predict(image)
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            start = time.time()
            self.predict(image)
            times.append(time.time() - start)
        
        times = np.array(times)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1.0 / np.mean(times)
        }
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': 'Faster R-CNN',
            'backbone': self.backbone,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }


if __name__ == "__main__":
    # Test the detector
    detector = FasterRCNNDetector(backbone='resnet50', device='cuda')
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Run prediction
    result = detector.predict(dummy_image)
    print(f"Number of detections: {result['num_detections']}")
    print(f"Inference time: {result['inference_time']:.4f}s")
    
    # Benchmark
    bench_results = detector.benchmark(dummy_image, num_iterations=50)
    print(f"Average FPS: {bench_results['fps']:.2f}")
