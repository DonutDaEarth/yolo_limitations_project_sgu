"""
Failure Case Analysis
Identify and visualize YOLO failure modes compared to Faster R-CNN
"""

import os
import sys
import numpy as np
import cv2
from typing import List, Dict, Tuple
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.data.dataset_loader import COCODatasetLoader
from src.evaluation.metrics import calculate_iou, calculate_box_area


class FailureCaseAnalyzer:
    """Analyze and identify failure cases"""
    
    def __init__(self, dataset_loader):
        """Initialize analyzer"""
        self.dataset_loader = dataset_loader
        self.failure_cases = []
    
    def find_false_negatives(self, yolo_preds, faster_rcnn_preds, ground_truths,
                            img_ids, min_iou=0.5) -> List[Dict]:
        """
        Find cases where YOLO misses objects that Faster R-CNN detects
        
        Args:
            yolo_preds: YOLO predictions
            faster_rcnn_preds: Faster R-CNN predictions
            ground_truths: Ground truth annotations
            img_ids: Image IDs
            min_iou: Minimum IoU to consider as detection
            
        Returns:
            List of failure cases
        """
        failure_cases = []
        
        print("Analyzing false negatives...")
        
        for idx, (yolo_pred, frcnn_pred, gt, img_id) in enumerate(
            zip(yolo_preds, faster_rcnn_preds, ground_truths, img_ids)):
            
            gt_boxes = gt['boxes']
            yolo_boxes = yolo_pred['boxes']
            frcnn_boxes = frcnn_pred['boxes']
            
            # For each ground truth object
            for gt_idx, gt_box in enumerate(gt_boxes):
                gt_area = calculate_box_area(gt_box)
                
                # Check if YOLO detected it
                yolo_detected = False
                yolo_max_iou = 0
                
                for yolo_box in yolo_boxes:
                    iou = calculate_iou(gt_box, yolo_box)
                    yolo_max_iou = max(yolo_max_iou, iou)
                    if iou >= min_iou:
                        yolo_detected = True
                        break
                
                # Check if Faster R-CNN detected it
                frcnn_detected = False
                frcnn_max_iou = 0
                
                for frcnn_box in frcnn_boxes:
                    iou = calculate_iou(gt_box, frcnn_box)
                    frcnn_max_iou = max(frcnn_max_iou, iou)
                    if iou >= min_iou:
                        frcnn_detected = True
                        break
                
                # If Faster R-CNN detected but YOLO didn't, it's a failure case
                if frcnn_detected and not yolo_detected:
                    # Determine object size category
                    if gt_area < 32 * 32:
                        size_category = 'small'
                    elif gt_area < 96 * 96:
                        size_category = 'medium'
                    else:
                        size_category = 'large'
                    
                    failure_cases.append({
                        'image_id': img_id,
                        'type': 'false_negative',
                        'gt_box': gt_box,
                        'gt_class': gt['classes'][gt_idx],
                        'gt_area': gt_area,
                        'size_category': size_category,
                        'yolo_max_iou': yolo_max_iou,
                        'frcnn_max_iou': frcnn_max_iou,
                        'severity': 'high' if frcnn_max_iou > 0.7 else 'medium'
                    })
        
        print(f"Found {len(failure_cases)} false negative cases")
        return failure_cases
    
    def find_poor_localization(self, yolo_preds, faster_rcnn_preds, ground_truths,
                               img_ids, iou_diff_threshold=0.15) -> List[Dict]:
        """
        Find cases where YOLO's localization is significantly worse than Faster R-CNN
        
        Args:
            yolo_preds: YOLO predictions
            faster_rcnn_preds: Faster R-CNN predictions  
            ground_truths: Ground truth annotations
            img_ids: Image IDs
            iou_diff_threshold: Minimum IoU difference to consider poor localization
            
        Returns:
            List of failure cases
        """
        failure_cases = []
        
        print("Analyzing poor localization...")
        
        for idx, (yolo_pred, frcnn_pred, gt, img_id) in enumerate(
            zip(yolo_preds, faster_rcnn_preds, ground_truths, img_ids)):
            
            gt_boxes = gt['boxes']
            yolo_boxes = yolo_pred['boxes']
            frcnn_boxes = frcnn_pred['boxes']
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                gt_area = calculate_box_area(gt_box)
                
                # Find best matching YOLO detection
                best_yolo_iou = 0
                best_yolo_box = None
                
                for yolo_box in yolo_boxes:
                    iou = calculate_iou(gt_box, yolo_box)
                    if iou > best_yolo_iou:
                        best_yolo_iou = iou
                        best_yolo_box = yolo_box
                
                # Find best matching Faster R-CNN detection
                best_frcnn_iou = 0
                best_frcnn_box = None
                
                for frcnn_box in frcnn_boxes:
                    iou = calculate_iou(gt_box, frcnn_box)
                    if iou > best_frcnn_iou:
                        best_frcnn_iou = iou
                        best_frcnn_box = frcnn_box
                
                # Check if Faster R-CNN's localization is significantly better
                if (best_frcnn_iou - best_yolo_iou) >= iou_diff_threshold:
                    # Determine size category
                    if gt_area < 32 * 32:
                        size_category = 'small'
                    elif gt_area < 96 * 96:
                        size_category = 'medium'
                    else:
                        size_category = 'large'
                    
                    failure_cases.append({
                        'image_id': img_id,
                        'type': 'poor_localization',
                        'gt_box': gt_box,
                        'yolo_box': best_yolo_box,
                        'frcnn_box': best_frcnn_box,
                        'gt_class': gt['classes'][gt_idx],
                        'gt_area': gt_area,
                        'size_category': size_category,
                        'yolo_iou': best_yolo_iou,
                        'frcnn_iou': best_frcnn_iou,
                        'iou_diff': best_frcnn_iou - best_yolo_iou,
                        'severity': 'high' if (best_frcnn_iou - best_yolo_iou) > 0.25 else 'medium'
                    })
        
        print(f"Found {len(failure_cases)} poor localization cases")
        return failure_cases
    
    def select_top_failures(self, failure_cases: List[Dict], top_k=10) -> List[Dict]:
        """
        Select top K failure cases based on severity and diversity
        
        Args:
            failure_cases: List of failure cases
            top_k: Number of cases to select
            
        Returns:
            Top K failure cases
        """
        # Sort by severity and size category
        sorted_cases = sorted(failure_cases, 
                            key=lambda x: (
                                x['severity'] == 'high',
                                x['size_category'] == 'small',
                                x.get('iou_diff', 0)
                            ),
                            reverse=True)
        
        # Select diverse cases
        selected = []
        size_counts = {'small': 0, 'medium': 0, 'large': 0}
        type_counts = {'false_negative': 0, 'poor_localization': 0}
        
        for case in sorted_cases:
            if len(selected) >= top_k:
                break
            
            # Ensure diversity
            size = case['size_category']
            case_type = case['type']
            
            # Prefer small objects and false negatives, but include variety
            if (size == 'small' or size_counts[size] < top_k // 3) and \
               (case_type == 'false_negative' or type_counts[case_type] < top_k // 2):
                selected.append(case)
                size_counts[size] += 1
                type_counts[case_type] += 1
        
        # Fill remaining slots if needed
        for case in sorted_cases:
            if len(selected) >= top_k:
                break
            if case not in selected:
                selected.append(case)
        
        return selected[:top_k]
    
    def visualize_failure_case(self, failure_case: Dict, save_path: str):
        """
        Visualize a single failure case
        
        Args:
            failure_case: Failure case dictionary
            save_path: Path to save visualization
        """
        img_id = failure_case['image_id']
        
        # Load image
        image, img_info = self.dataset_loader.load_image(img_id)
        
        # Create visualization with three panels
        h, w = image.shape[:2]
        canvas = np.ones((h, w * 3 + 40, 3), dtype=np.uint8) * 255
        
        # Ground truth panel
        gt_image = image.copy()
        gt_box = failure_case['gt_box'].astype(int)
        cv2.rectangle(gt_image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]),
                     (0, 255, 0), 3)
        canvas[:, :w] = gt_image
        
        # YOLO panel
        yolo_image = image.copy()
        if failure_case['type'] == 'poor_localization' and failure_case.get('yolo_box') is not None:
            yolo_box = failure_case['yolo_box'].astype(int)
            cv2.rectangle(yolo_image, (yolo_box[0], yolo_box[1]), 
                         (yolo_box[2], yolo_box[3]), (0, 0, 255), 3)
        # Draw ground truth in light color for reference
        cv2.rectangle(yolo_image, (gt_box[0], gt_box[1]), (gt_box[2], gt_box[3]),
                     (0, 255, 0), 1)
        canvas[:, w+20:w*2+20] = yolo_image
        
        # Faster R-CNN panel
        frcnn_image = image.copy()
        if failure_case.get('frcnn_box') is not None:
            frcnn_box = failure_case['frcnn_box'].astype(int)
            cv2.rectangle(frcnn_image, (frcnn_box[0], frcnn_box[1]),
                         (frcnn_box[2], frcnn_box[3]), (255, 0, 0), 3)
        canvas[:, w*2+40:] = frcnn_image
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, 'Ground Truth', (w//2-80, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(canvas, 'YOLO (Failed)', (w+w//2-60, 30), font, 1, (0, 0, 0), 2)
        cv2.putText(canvas, 'Faster R-CNN (Success)', (w*2+40+w//2-120, 30), 
                   font, 1, (0, 0, 0), 2)
        
        # Add info
        info_text = f"Type: {failure_case['type']} | Size: {failure_case['size_category']} | " \
                   f"Area: {failure_case['gt_area']:.0f}px²"
        cv2.putText(canvas, info_text, (20, h-20), font, 0.6, (0, 0, 0), 1)
        
        # Save
        cv2.imwrite(save_path, canvas)
        print(f"Saved visualization: {save_path}")


def main():
    """Main execution"""
    # Paths
    RESULTS_DIR = "./results/metrics"
    FAILURE_DIR = "./results/failure_cases"
    os.makedirs(FAILURE_DIR, exist_ok=True)
    
    # Load predictions
    pred_file = os.path.join(RESULTS_DIR, 'predictions.npz')
    
    if not os.path.exists(pred_file):
        print("❌ Error: Predictions file not found!")
        print("   Please run Task A first: python scripts/run_taskA.py")
        return
    
    print("Loading predictions...")
    data = np.load(pred_file, allow_pickle=True)
    yolo_preds = data['yolo_preds']
    yolo_gts = data['yolo_gts']
    faster_rcnn_preds = data['faster_rcnn_preds']
    img_ids = data['img_ids']
    
    # Load dataset
    print("Loading dataset...")
    dataset_loader = COCODatasetLoader(
        data_root="./data/coco",
        split='val',
        year=2017
    )
    
    # Create analyzer
    analyzer = FailureCaseAnalyzer(dataset_loader)
    
    # Find failure cases
    print("\n" + "="*80)
    print("ANALYZING FAILURE CASES")
    print("="*80)
    
    false_negatives = analyzer.find_false_negatives(
        yolo_preds, faster_rcnn_preds, yolo_gts, img_ids
    )
    
    poor_localizations = analyzer.find_poor_localization(
        yolo_preds, faster_rcnn_preds, yolo_gts, img_ids
    )
    
    # Combine and select top cases
    all_failures = false_negatives + poor_localizations
    top_failures = analyzer.select_top_failures(all_failures, top_k=10)
    
    # Helper function to convert numpy types to Python types
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    # Save failure case data
    failure_data_file = os.path.join(FAILURE_DIR, 'failure_cases.json')
    with open(failure_data_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_failures = [convert_numpy_types(case) for case in top_failures]
        json.dump(json_failures, f, indent=4)
    
    print(f"\nFailure case data saved to {failure_data_file}")
    
    # Visualize top failures
    print("\nGenerating visualizations...")
    for i, failure in enumerate(top_failures, 1):
        save_path = os.path.join(FAILURE_DIR, f'failure_case_{i:02d}.png')
        analyzer.visualize_failure_case(failure, save_path)
    
    # Print summary
    print("\n" + "="*80)
    print("FAILURE CASE SUMMARY")
    print("="*80)
    
    # Statistics by type
    fn_count = sum(1 for f in top_failures if f['type'] == 'false_negative')
    pl_count = sum(1 for f in top_failures if f['type'] == 'poor_localization')
    
    print(f"\nTotal failure cases found: {len(all_failures)}")
    print(f"  - False Negatives: {len(false_negatives)}")
    print(f"  - Poor Localizations: {len(poor_localizations)}")
    
    print(f"\nTop {len(top_failures)} cases selected:")
    print(f"  - False Negatives: {fn_count}")
    print(f"  - Poor Localizations: {pl_count}")
    
    # Statistics by size
    size_counts = {'small': 0, 'medium': 0, 'large': 0}
    for f in top_failures:
        size_counts[f['size_category']] += 1
    
    print(f"\nBy object size:")
    print(f"  - Small objects: {size_counts['small']}")
    print(f"  - Medium objects: {size_counts['medium']}")
    print(f"  - Large objects: {size_counts['large']}")
    
    print("\n✓ Failure case analysis completed!")
    print(f"  Visualizations saved in: {FAILURE_DIR}/")
    print(f"  Use these images in your presentation and report.")


if __name__ == "__main__":
    main()
