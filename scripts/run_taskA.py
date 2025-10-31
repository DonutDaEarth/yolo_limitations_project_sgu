"""
Task A: Small Object & Clutter Challenge
Run inference and calculate metrics for both models
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.yolo_detector import YOLODetector
from src.models.two_stage_detector import FasterRCNNDetector
from src.data.dataset_loader import COCODatasetLoader
from src.evaluation.metrics import calculate_map_coco


def run_inference(detector, dataset_loader, img_ids):
    """Run inference on dataset"""
    predictions = []
    ground_truths = []
    
    print(f"Running inference on {len(img_ids)} images...")
    
    for img_id in tqdm(img_ids):
        # Load image and annotations
        image, _ = dataset_loader.load_image(img_id)
        annotations = dataset_loader.load_annotations(img_id)
        
        # Run inference
        pred = detector.predict(image)
        
        # Store predictions
        predictions.append({
            'boxes': pred['boxes'],
            'scores': pred['scores'],
            'classes': pred['classes'],
            'image_id': img_id
        })
        
        # Store ground truths
        gt_boxes = np.array([ann['bbox'] for ann in annotations])
        gt_classes = np.array([ann['category_id'] for ann in annotations])
        
        ground_truths.append({
            'boxes': gt_boxes,
            'classes': gt_classes,
            'image_id': img_id
        })
    
    return predictions, ground_truths


def main():
    parser = argparse.ArgumentParser(description='Run Task A: Small Object Challenge')
    parser.add_argument('--dataset', type=str, default='coco', help='Dataset name')
    parser.add_argument('--split', type=str, default='val', help='Dataset split')
    parser.add_argument('--num_images', type=int, default=500, help='Number of images')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--small_objects_only', action='store_true', 
                       help='Use only images with small objects')
    parser.add_argument('--ann_file', type=str, default=None,
                       help='Custom annotation file (e.g., subset file)')
    
    args = parser.parse_args()
    
    # Configuration
    DATA_ROOT = "./data/coco"
    RESULTS_DIR = "./results/metrics"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Auto-detect subset annotation file if num_images <= 500 and no custom file specified
    if args.ann_file is None and args.num_images <= 500:
        subset_file = f"./data/coco/annotations/instances_val2017_subset{args.num_images}.json"
        if os.path.exists(subset_file):
            print(f"Using subset annotation file: {subset_file}")
            args.ann_file = subset_file
    
    # Load dataset
    print("Loading dataset...")
    dataset_loader = COCODatasetLoader(
        data_root=DATA_ROOT,
        split=args.split,
        year=2017,
        annotations_file=args.ann_file
    )
    
    # Get image IDs
    if args.small_objects_only:
        print("Selecting images with small objects...")
        img_ids = dataset_loader.get_small_object_subset(max_images=args.num_images)
    else:
        img_ids = dataset_loader.get_image_ids()[:args.num_images]
    
    print(f"Using {len(img_ids)} images for evaluation")
    
    # Initialize models
    print("\nInitializing models...")
    yolo = YOLODetector(model_variant='yolov8n', device=args.device)
    faster_rcnn = FasterRCNNDetector(backbone='resnet50', device=args.device)
    
    # Run inference for YOLO
    print("\n" + "="*80)
    print("YOLO INFERENCE")
    print("="*80)
    yolo_preds, yolo_gts = run_inference(yolo, dataset_loader, img_ids)
    
    # Calculate YOLO metrics
    print("\nCalculating YOLO metrics...")
    yolo_metrics = calculate_map_coco(yolo_preds, yolo_gts)
    
    # Run inference for Faster R-CNN
    print("\n" + "="*80)
    print("FASTER R-CNN INFERENCE")
    print("="*80)
    faster_rcnn_preds, faster_rcnn_gts = run_inference(faster_rcnn, dataset_loader, img_ids)
    
    # Calculate Faster R-CNN metrics
    print("\nCalculating Faster R-CNN metrics...")
    faster_rcnn_metrics = calculate_map_coco(faster_rcnn_preds, faster_rcnn_gts)
    
    # Save results
    results = {
        'yolo': {
            'model_info': yolo.get_model_info(),
            'metrics': yolo_metrics,
            'num_images': len(img_ids)
        },
        'faster_rcnn': {
            'model_info': faster_rcnn.get_model_info(),
            'metrics': faster_rcnn_metrics,
            'num_images': len(img_ids)
        }
    }
    
    # Save to JSON
    output_file = os.path.join(RESULTS_DIR, 'taskA_results.json')
    with open(output_file, 'w') as f:
        # Convert numpy types to Python types
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'model_info': result['model_info'],
                'metrics': {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in result['metrics'].items()
                },
                'num_images': result['num_images']
            }
        json.dump(json_results, f, indent=4)
    
    print(f"\nResults saved to {output_file}")
    
    # Save predictions for failure case analysis
    pred_output = os.path.join(RESULTS_DIR, 'predictions.npz')
    np.savez(pred_output,
             yolo_preds=yolo_preds,
             yolo_gts=yolo_gts,
             faster_rcnn_preds=faster_rcnn_preds,
             faster_rcnn_gts=faster_rcnn_gts,
             img_ids=img_ids)
    print(f"Predictions saved to {pred_output}")
    
    # Print comparison
    print("\n" + "="*80)
    print("METRICS COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<20} {'YOLO':<15} {'Faster R-CNN':<15} {'Difference':<15}")
    print("-"*65)
    
    for metric in ['mAP@[0.5:0.95]', 'mAP@0.5', 'mAP(Small)', 'mAP(Medium)', 'mAP(Large)']:
        yolo_val = yolo_metrics[metric]
        frcnn_val = faster_rcnn_metrics[metric]
        diff = frcnn_val - yolo_val
        
        print(f"{metric:<20} {yolo_val*100:>6.2f}% {frcnn_val*100:>14.2f}% "
              f"{diff*100:>+14.2f}%")
    
    print("="*80)
    
    # Highlight key findings
    print("\nKEY FINDINGS:")
    print("-" * 80)
    
    small_diff = (faster_rcnn_metrics['mAP(Small)'] - yolo_metrics['mAP(Small)']) * 100
    if small_diff > 5:
        print(f"⚠️  Faster R-CNN outperforms YOLO on small objects by {small_diff:.2f}%")
        print("   This demonstrates YOLO's limitations in detecting small objects.")
    
    overall_diff = (faster_rcnn_metrics['mAP@[0.5:0.95]'] - yolo_metrics['mAP@[0.5:0.95]']) * 100
    if overall_diff > 2:
        print(f"⚠️  Faster R-CNN has {overall_diff:.2f}% higher overall mAP")
        print("   This shows the accuracy trade-off of single-shot detectors.")
    
    print("\n✓ Task A completed successfully!")
    print("  Next steps:")
    print("  1. Run Task B for speed benchmarking: python scripts/run_taskB.py")
    print("  2. Analyze failure cases: python src/visualization/failure_cases.py")


if __name__ == "__main__":
    main()
