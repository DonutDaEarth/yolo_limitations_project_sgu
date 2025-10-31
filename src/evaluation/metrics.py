"""
Evaluation Metrics for Object Detection
Implements mAP, mAP(Small), and other metrics
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two bounding boxes
    
    Args:
        box1: Box in format [x1, y1, x2, y2]
        box2: Box in format [x1, y1, x2, y2]
        
    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_box_area(box: np.ndarray) -> float:
    """Calculate area of bounding box"""
    return (box[2] - box[0]) * (box[3] - box[1])


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision using 11-point interpolation
    
    Args:
        recall: Recall values
        precision: Precision values
        
    Returns:
        Average Precision
    """
    # Add sentinel values
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute the precision envelope
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    
    # Calculate area under curve
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    
    return ap


def calculate_map(predictions: List[Dict], ground_truths: List[Dict],
                  iou_threshold: float = 0.5, num_classes: int = 80) -> Dict:
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of prediction dicts with keys: boxes, scores, classes
        ground_truths: List of ground truth dicts with keys: boxes, classes
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
        
    Returns:
        Dictionary with mAP and per-class AP
    """
    aps = []
    
    for class_id in range(num_classes):
        # Collect all predictions and ground truths for this class
        class_predictions = []
        class_gts = []
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Get predictions for this class
            pred_mask = pred['classes'] == class_id
            if np.any(pred_mask):
                pred_boxes = pred['boxes'][pred_mask]
                pred_scores = pred['scores'][pred_mask]
                
                for box, score in zip(pred_boxes, pred_scores):
                    class_predictions.append({
                        'image_id': img_idx,
                        'box': box,
                        'score': score
                    })
            
            # Get ground truths for this class
            gt_mask = gt['classes'] == class_id
            if np.any(gt_mask):
                gt_boxes = gt['boxes'][gt_mask]
                
                class_gts.append({
                    'image_id': img_idx,
                    'boxes': gt_boxes,
                    'detected': np.zeros(len(gt_boxes), dtype=bool)
                })
        
        if len(class_predictions) == 0:
            continue
        
        # Sort predictions by confidence
        class_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate precision and recall
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        
        total_gts = sum(len(gt['boxes']) for gt in class_gts)
        
        for pred_idx, pred in enumerate(class_predictions):
            img_id = pred['image_id']
            pred_box = pred['box']
            
            # Find corresponding ground truths
            img_gts = [gt for gt in class_gts if gt['image_id'] == img_id]
            
            if len(img_gts) == 0:
                fp[pred_idx] = 1
                continue
            
            img_gt = img_gts[0]
            gt_boxes = img_gt['boxes']
            detected = img_gt['detected']
            
            # Find best matching ground truth
            max_iou = 0
            max_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if detected[gt_idx]:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = gt_idx
            
            # Check if match is valid
            if max_iou >= iou_threshold:
                if not detected[max_idx]:
                    tp[pred_idx] = 1
                    detected[max_idx] = True
                else:
                    fp[pred_idx] = 1
            else:
                fp[pred_idx] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / total_gts if total_gts > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap = compute_ap(recalls, precisions)
        aps.append(ap)
    
    # Calculate mAP
    map_score = np.mean(aps) if len(aps) > 0 else 0.0
    
    return {
        'mAP': map_score,
        'num_classes': len(aps),
        'per_class_ap': aps
    }


def calculate_map_coco(predictions: List[Dict], ground_truths: List[Dict],
                       iou_thresholds: List[float] = None) -> Dict:
    """
    Calculate mAP using COCO evaluation protocol
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        iou_thresholds: List of IoU thresholds (default: 0.5:0.05:0.95)
        
    Returns:
        Dictionary with mAP@[0.5:0.95], mAP@0.5, mAP(Small), etc.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
    
    # Calculate mAP for each IoU threshold
    maps = []
    for iou_thresh in iou_thresholds:
        result = calculate_map(predictions, ground_truths, iou_threshold=iou_thresh)
        maps.append(result['mAP'])
    
    # mAP@[0.5:0.95] is the average over all IoU thresholds
    map_50_95 = np.mean(maps)
    
    # mAP@0.5
    map_50 = calculate_map(predictions, ground_truths, iou_threshold=0.5)['mAP']
    
    # mAP for different object sizes
    small_preds, small_gts = filter_by_size(predictions, ground_truths, 
                                             min_area=0, max_area=32*32)
    map_small = calculate_map(small_preds, small_gts, iou_threshold=0.5)['mAP']
    
    medium_preds, medium_gts = filter_by_size(predictions, ground_truths, 
                                               min_area=32*32, max_area=96*96)
    map_medium = calculate_map(medium_preds, medium_gts, iou_threshold=0.5)['mAP']
    
    large_preds, large_gts = filter_by_size(predictions, ground_truths, 
                                             min_area=96*96)
    map_large = calculate_map(large_preds, large_gts, iou_threshold=0.5)['mAP']
    
    return {
        'mAP@[0.5:0.95]': map_50_95,
        'mAP@0.5': map_50,
        'mAP(Small)': map_small,
        'mAP(Medium)': map_medium,
        'mAP(Large)': map_large
    }


def filter_by_size(predictions: List[Dict], ground_truths: List[Dict],
                   min_area: float = 0, max_area: float = float('inf')) -> Tuple:
    """
    Filter predictions and ground truths by object size
    
    Args:
        predictions: List of predictions
        ground_truths: List of ground truths
        min_area: Minimum area
        max_area: Maximum area
        
    Returns:
        Tuple of (filtered_predictions, filtered_ground_truths)
    """
    filtered_preds = []
    filtered_gts = []
    
    for pred, gt in zip(predictions, ground_truths):
        # Filter predictions
        pred_areas = np.array([calculate_box_area(box) for box in pred['boxes']])
        pred_mask = (pred_areas >= min_area) & (pred_areas < max_area)
        
        filtered_preds.append({
            'boxes': pred['boxes'][pred_mask],
            'scores': pred['scores'][pred_mask],
            'classes': pred['classes'][pred_mask]
        })
        
        # Filter ground truths
        gt_areas = np.array([calculate_box_area(box) for box in gt['boxes']])
        gt_mask = (gt_areas >= min_area) & (gt_areas < max_area)
        
        filtered_gts.append({
            'boxes': gt['boxes'][gt_mask],
            'classes': gt['classes'][gt_mask]
        })
    
    return filtered_preds, filtered_gts


if __name__ == "__main__":
    # Test metrics
    # Create dummy predictions and ground truths
    predictions = [
        {
            'boxes': np.array([[10, 10, 50, 50], [60, 60, 100, 100]]),
            'scores': np.array([0.9, 0.8]),
            'classes': np.array([0, 1])
        }
    ]
    
    ground_truths = [
        {
            'boxes': np.array([[12, 12, 52, 52], [58, 58, 98, 98]]),
            'classes': np.array([0, 1])
        }
    ]
    
    # Calculate mAP
    result = calculate_map_coco(predictions, ground_truths)
    print("mAP Results:")
    for key, value in result.items():
        print(f"  {key}: {value:.4f}")
