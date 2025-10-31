"""
YOLO Detector Wrapper
Provides a unified interface for YOLOv8 models
"""

import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
import time


class YOLODetector:
    """Wrapper class for YOLO models with consistent interface"""
    
    def __init__(self, model_variant='yolov8n', weights=None, conf_threshold=0.25, 
                 iou_threshold=0.45, device='cuda'):
        """
        Initialize YOLO detector
        
        Args:
            model_variant: YOLO variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
            weights: Path to weights file or model name
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.model_variant = model_variant
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        if weights is None:
            weights = f'{model_variant}.pt'
        
        print(f"Loading YOLO model: {weights}")
        self.model = YOLO(weights)
        
        # Move to device
        self.model.to(device)
        
        print(f"YOLO {model_variant} loaded successfully on {device}")
    
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
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold
        
        # Measure inference time
        start_time = time.time()
        
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            verbose=False,
            device=self.device
        )
        
        inference_time = time.time() - start_time
        
        # Extract predictions
        result = results[0]  # Single image
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        
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
            'model_type': 'YOLO',
            'variant': self.model_variant,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }


if __name__ == "__main__":
    # Test the detector
    detector = YOLODetector(model_variant='yolov8n', device='cuda')
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Run prediction
    result = detector.predict(dummy_image)
    print(f"Number of detections: {result['num_detections']}")
    print(f"Inference time: {result['inference_time']:.4f}s")
    
    # Benchmark
    bench_results = detector.benchmark(dummy_image, num_iterations=50)
    print(f"Average FPS: {bench_results['fps']:.2f}")
