"""
Speed Benchmark Script for Task B
Measures FPS and creates speed-accuracy trade-off visualization
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json
import os
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.yolo_detector import YOLODetector
from models.two_stage_detector import FasterRCNNDetector
from data.dataset_loader import COCODatasetLoader


class SpeedBenchmark:
    """Benchmark inference speed for object detectors"""
    
    def __init__(self, device='cuda'):
        """Initialize benchmark"""
        self.device = device
        self.results = {}
    
    def benchmark_model(self, detector, dataset_loader, num_images=500, 
                        warmup=10) -> Dict:
        """
        Benchmark a detector model
        
        Args:
            detector: Detector instance (YOLODetector or FasterRCNNDetector)
            dataset_loader: Dataset loader instance
            num_images: Number of images to benchmark on
            warmup: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"\nBenchmarking {detector.get_model_info()['model_type']}...")
        
        # Get image IDs
        img_ids = dataset_loader.get_image_ids()[:num_images]
        
        # Warmup
        print("Warming up...")
        for i in range(min(warmup, len(img_ids))):
            image, _ = dataset_loader.load_image(img_ids[i])
            detector.predict(image)
        
        # Benchmark
        print(f"Running benchmark on {len(img_ids)} images...")
        inference_times = []
        num_detections = []
        
        for img_id in tqdm(img_ids):
            image, _ = dataset_loader.load_image(img_id)
            
            # Measure inference time
            start_time = time.time()
            result = detector.predict(image)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            num_detections.append(result['num_detections'])
        
        inference_times = np.array(inference_times)
        
        # Calculate statistics
        results = {
            'mean_time': np.mean(inference_times),
            'std_time': np.std(inference_times),
            'median_time': np.median(inference_times),
            'min_time': np.min(inference_times),
            'max_time': np.max(inference_times),
            'fps': 1.0 / np.mean(inference_times),
            'total_time': np.sum(inference_times),
            'num_images': len(img_ids),
            'avg_detections': np.mean(num_detections),
            'model_info': detector.get_model_info()
        }
        
        return results
    
    def create_speed_accuracy_plot(self, results_dict: Dict[str, Dict], 
                                    save_path: str = None):
        """
        Create speed-accuracy trade-off plot
        
        Args:
            results_dict: Dictionary mapping model names to their results
                         Each result should contain 'fps' and 'mAP@[0.5:0.95]'
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data
        model_names = []
        fps_values = []
        map_values = []
        
        for model_name, results in results_dict.items():
            model_names.append(model_name)
            fps_values.append(results['fps'])
            map_values.append(results.get('mAP@[0.5:0.95]', 0) * 100)  # Convert to percentage
        
        # Create scatter plot
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        for i, (name, fps, map_val) in enumerate(zip(model_names, fps_values, map_values)):
            plt.scatter(fps, map_val, s=200, alpha=0.7, 
                       color=colors[i % len(colors)], label=name)
            plt.annotate(name, (fps, map_val), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        # Styling
        plt.xlabel('Frames Per Second (FPS)', fontsize=12, fontweight='bold')
        plt.ylabel('mAP@[0.5:0.95] (%)', fontsize=12, fontweight='bold')
        plt.title('Speed-Accuracy Trade-off: YOLO vs Faster R-CNN', 
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='lower right')
        
        # Add diagonal reference lines
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def create_detailed_comparison_table(self, results_dict: Dict[str, Dict]):
        """
        Create detailed comparison table
        
        Args:
            results_dict: Dictionary mapping model names to their results
        """
        print("\n" + "="*80)
        print("DETAILED COMPARISON TABLE")
        print("="*80)
        
        # Headers
        headers = ["Metric", "Unit"]
        for model_name in results_dict.keys():
            headers.append(model_name)
        
        # Print headers
        print(f"{headers[0]:<30} {headers[1]:<10}", end="")
        for h in headers[2:]:
            print(f"{h:>15}", end="")
        print()
        print("-"*80)
        
        # Metrics to display
        metrics = [
            ('fps', 'FPS', '{:.2f}'),
            ('mean_time', 'ms', '{:.2f}'),
            ('mAP@[0.5:0.95]', '%', '{:.2f}'),
            ('mAP@0.5', '%', '{:.2f}'),
            ('mAP(Small)', '%', '{:.2f}'),
            ('total_time', 's', '{:.2f}')
        ]
        
        for metric_key, unit, fmt in metrics:
            print(f"{metric_key:<30} {unit:<10}", end="")
            
            for model_name in results_dict.keys():
                value = results_dict[model_name].get(metric_key, 0)
                
                # Convert time to ms if needed
                if unit == 'ms' and metric_key == 'mean_time':
                    value *= 1000
                
                # Convert to percentage if needed
                if unit == '%' and value < 1:
                    value *= 100
                
                print(f"{fmt.format(value):>15}", end="")
            print()
        
        print("="*80)


def main():
    """Main benchmark script"""
    # Configuration
    DATA_ROOT = "./data/coco"
    NUM_IMAGES = 500
    WARMUP = 10
    DEVICE = 'cuda'
    RESULTS_DIR = "./results/benchmark"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load dataset
    print("Loading COCO dataset...")
    dataset_loader = COCODatasetLoader(
        data_root=DATA_ROOT,
        split='val',
        year=2017
    )
    
    # Initialize models
    print("\nInitializing models...")
    yolo = YOLODetector(model_variant='yolov8n', device=DEVICE)
    faster_rcnn = FasterRCNNDetector(backbone='resnet50', device=DEVICE)
    
    # Create benchmark instance
    benchmark = SpeedBenchmark(device=DEVICE)
    
    # Benchmark YOLO
    yolo_results = benchmark.benchmark_model(
        yolo, dataset_loader, num_images=NUM_IMAGES, warmup=WARMUP
    )
    
    # Benchmark Faster R-CNN
    faster_rcnn_results = benchmark.benchmark_model(
        faster_rcnn, dataset_loader, num_images=NUM_IMAGES, warmup=WARMUP
    )
    
    # Save results
    results = {
        'YOLOv8n': yolo_results,
        'Faster R-CNN': faster_rcnn_results
    }
    
    # Save to JSON
    output_file = os.path.join(RESULTS_DIR, 'speed_benchmark_results.json')
    with open(output_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in result.items() if k != 'model_info'
            }
        json.dump(json_results, f, indent=4)
    
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SPEED BENCHMARK SUMMARY")
    print("="*80)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  FPS: {result['fps']:.2f}")
        print(f"  Mean inference time: {result['mean_time']*1000:.2f} ms")
        print(f"  Total time: {result['total_time']:.2f} s")
        print(f"  Average detections: {result['avg_detections']:.1f}")
    
    print("\n" + "="*80)
    
    # Note: mAP values need to be added from Task A results
    print("\nNote: To create the speed-accuracy plot, run Task A first to get mAP values.")
    print("Then update the results with mAP values and run the visualization.")


if __name__ == "__main__":
    main()
