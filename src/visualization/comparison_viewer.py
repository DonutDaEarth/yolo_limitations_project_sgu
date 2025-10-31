"""
Side-by-Side Comparison Viewer
Interactive tool to compare YOLO and Faster R-CNN detections
"""

import os
import sys
import numpy as np
import cv2
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.models.yolo_detector import YOLODetector
from src.models.two_stage_detector import FasterRCNNDetector
from src.data.dataset_loader import COCODatasetLoader


class ComparisonViewer:
    """Interactive viewer for comparing detections"""
    
    def __init__(self, yolo_detector, frcnn_detector, dataset_loader):
        """Initialize viewer"""
        self.yolo = yolo_detector
        self.frcnn = frcnn_detector
        self.dataset = dataset_loader
        self.current_idx = 0
        self.img_ids = None
    
    def draw_detections(self, image, boxes, scores, classes, color, label_prefix=""):
        """Draw bounding boxes on image"""
        img_copy = image.copy()
        
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{label_prefix}Class {cls}: {score:.2f}"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1, y1 - text_h - 5), (x1 + text_w, y1), color, -1)
            
            # Text
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_copy
    
    def create_comparison_view(self, image, yolo_pred, frcnn_pred, gt_boxes=None):
        """Create side-by-side comparison"""
        h, w = image.shape[:2]
        
        # Create canvas
        canvas = np.ones((h + 100, w * 2 + 30, 3), dtype=np.uint8) * 255
        
        # Draw YOLO detections
        yolo_image = self.draw_detections(
            image, yolo_pred['boxes'], yolo_pred['scores'], 
            yolo_pred['classes'], (0, 0, 255), "YOLO "
        )
        
        # Draw Faster R-CNN detections
        frcnn_image = self.draw_detections(
            image, frcnn_pred['boxes'], frcnn_pred['scores'],
            frcnn_pred['classes'], (255, 0, 0), "FRCNN "
        )
        
        # Draw ground truth on both if available
        if gt_boxes is not None:
            for box in gt_boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(yolo_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.rectangle(frcnn_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Place images on canvas
        canvas[50:50+h, 10:10+w] = yolo_image
        canvas[50:50+h, 20+w:20+w*2] = frcnn_image
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, 'YOLO Detection', (w//2-100, 35), font, 1, (0, 0, 255), 2)
        cv2.putText(canvas, 'Faster R-CNN Detection', (w+20+w//2-150, 35), font, 1, (255, 0, 0), 2)
        
        # Add statistics
        stats_y = h + 70
        cv2.putText(canvas, f"YOLO: {len(yolo_pred['boxes'])} detections, "
                   f"Avg conf: {np.mean(yolo_pred['scores']):.2f}",
                   (20, stats_y), font, 0.6, (0, 0, 0), 1)
        
        cv2.putText(canvas, f"Faster R-CNN: {len(frcnn_pred['boxes'])} detections, "
                   f"Avg conf: {np.mean(frcnn_pred['scores']):.2f}",
                   (w+30, stats_y), font, 0.6, (0, 0, 0), 1)
        
        # Add legend
        legend_y = h + 90
        cv2.rectangle(canvas, (20, legend_y-10), (35, legend_y-5), (0, 255, 0), -1)
        cv2.putText(canvas, "= Ground Truth", (40, legend_y), font, 0.5, (0, 0, 0), 1)
        
        return canvas
    
    def run_interactive(self, img_ids, save_dir=None):
        """Run interactive viewer"""
        self.img_ids = img_ids
        
        print("\n" + "="*80)
        print("INTERACTIVE COMPARISON VIEWER")
        print("="*80)
        print("\nControls:")
        print("  n - Next image")
        print("  p - Previous image")
        print("  s - Save current comparison")
        print("  q - Quit")
        print("\nPress any key to start...")
        
        while True:
            # Load image
            img_id = self.img_ids[self.current_idx]
            image, img_info = self.dataset.load_image(img_id)
            annotations = self.dataset.load_annotations(img_id)
            
            # Get predictions
            print(f"\nProcessing image {self.current_idx + 1}/{len(self.img_ids)}: "
                  f"{img_info['file_name']}")
            
            yolo_pred = self.yolo.predict(image)
            frcnn_pred = self.frcnn.predict(image)
            
            # Ground truth boxes
            gt_boxes = np.array([ann['bbox'] for ann in annotations]) if annotations else None
            
            # Create comparison
            comparison = self.create_comparison_view(image, yolo_pred, frcnn_pred, gt_boxes)
            
            # Show
            cv2.imshow('YOLO vs Faster R-CNN Comparison', comparison)
            
            # Wait for key
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('n'):
                self.current_idx = (self.current_idx + 1) % len(self.img_ids)
            elif key == ord('p'):
                self.current_idx = (self.current_idx - 1) % len(self.img_ids)
            elif key == ord('s'):
                if save_dir:
                    save_path = os.path.join(save_dir, f'comparison_{img_id:012d}.png')
                    cv2.imwrite(save_path, comparison)
                    print(f"Saved: {save_path}")
        
        cv2.destroyAllWindows()
    
    def generate_comparisons(self, img_ids, save_dir, max_images=20):
        """Generate and save comparisons for multiple images"""
        print(f"\nGenerating comparisons for {min(len(img_ids), max_images)} images...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, img_id in enumerate(img_ids[:max_images]):
            print(f"Processing {idx+1}/{min(len(img_ids), max_images)}...")
            
            # Load image
            image, img_info = self.dataset.load_image(img_id)
            annotations = self.dataset.load_annotations(img_id)
            
            # Get predictions
            yolo_pred = self.yolo.predict(image)
            frcnn_pred = self.frcnn.predict(image)
            
            # Ground truth
            gt_boxes = np.array([ann['bbox'] for ann in annotations]) if annotations else None
            
            # Create comparison
            comparison = self.create_comparison_view(image, yolo_pred, frcnn_pred, gt_boxes)
            
            # Save
            save_path = os.path.join(save_dir, f'comparison_{idx+1:03d}.png')
            cv2.imwrite(save_path, comparison)
        
        print(f"\n✓ Saved {min(len(img_ids), max_images)} comparisons to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Side-by-Side Comparison Viewer')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run interactive viewer')
    parser.add_argument('--generate', action='store_true',
                       help='Generate comparison images')
    parser.add_argument('--num_images', type=int, default=20,
                       help='Number of images to process')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Configuration
    DATA_ROOT = "./data/coco"
    SAVE_DIR = "./results/comparisons"
    
    # Load dataset
    print("Loading dataset...")
    dataset_loader = COCODatasetLoader(
        data_root=DATA_ROOT,
        split='val',
        year=2017
    )
    
    # Get interesting images (with small objects)
    print("Selecting images with small objects...")
    img_ids = dataset_loader.get_small_object_subset(max_images=args.num_images)
    
    # Initialize models
    print("Loading models...")
    yolo = YOLODetector(model_variant='yolov8n', device=args.device)
    frcnn = FasterRCNNDetector(backbone='resnet50', device=args.device)
    
    # Create viewer
    viewer = ComparisonViewer(yolo, frcnn, dataset_loader)
    
    if args.interactive:
        # Run interactive mode
        viewer.run_interactive(img_ids, save_dir=SAVE_DIR)
    elif args.generate:
        # Generate comparisons
        viewer.generate_comparisons(img_ids, SAVE_DIR, max_images=args.num_images)
    else:
        print("\nPlease specify --interactive or --generate mode")
        print("  python src/visualization/comparison_viewer.py --interactive")
        print("  python src/visualization/comparison_viewer.py --generate --num_images 20")


if __name__ == "__main__":
    main()
