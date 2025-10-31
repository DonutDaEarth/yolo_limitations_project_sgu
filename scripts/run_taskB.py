"""
Task B: Speed vs Accuracy Trade-off
Benchmark inference speed and create visualization
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.yolo_detector import YOLODetector
from src.models.two_stage_detector import FasterRCNNDetector
from src.data.dataset_loader import COCODatasetLoader
from src.evaluation.speed_benchmark import SpeedBenchmark


def create_speed_accuracy_plot(results, save_path):
    """Create the speed-accuracy trade-off plot"""
    plt.figure(figsize=(12, 7))
    
    # Extract data
    models = list(results.keys())
    fps_values = [results[m]['fps'] for m in models]
    map_values = [results[m].get('mAP@[0.5:0.95]', 0) * 100 for m in models]
    
    # Color scheme
    colors = {
        'YOLOv8n': '#FF6B6B',
        'YOLOv8s': '#FF8E53',
        'YOLOv8m': '#FFA726',
        'Faster R-CNN (R50)': '#4ECDC4',
        'Faster R-CNN (R101)': '#45B7D1'
    }
    
    # Plot points
    for model in models:
        fps = results[model]['fps']
        map_val = results[model].get('mAP@[0.5:0.95]', 0) * 100
        color = colors.get(model, '#95E1D3')
        
        plt.scatter(fps, map_val, s=300, alpha=0.7, color=color, 
                   edgecolors='black', linewidth=2, label=model, zorder=3)
        
        # Add annotations with arrows
        offset_x = 15 if 'YOLO' in model else -80
        offset_y = 10 if 'YOLO' in model else -15
        
        plt.annotate(model, 
                    xy=(fps, map_val),
                    xytext=(offset_x, offset_y),
                    textcoords='offset points',
                    fontsize=11,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                                   lw=1.5))
    
    # Styling
    plt.xlabel('Frames Per Second (FPS)', fontsize=14, fontweight='bold')
    plt.ylabel('mAP@[0.5:0.95] (%)', fontsize=14, fontweight='bold')
    plt.title('Speed-Accuracy Trade-off: Single-Shot vs Two-Stage Detectors',
             fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    
    # Add regions
    ax = plt.gca()
    
    # High speed, lower accuracy region
    ax.axvspan(max(fps_values)*0.7, max(fps_values)*1.1, alpha=0.1, color='red',
              label='High Speed Region')
    
    # Lower speed, high accuracy region  
    ax.axvspan(0, min(fps_values)*1.3, alpha=0.1, color='green',
              label='High Accuracy Region')
    
    # Style axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Legend
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    
    return plt


def main():
    parser = argparse.ArgumentParser(description='Run Task B: Speed Benchmark')
    parser.add_argument('--num_images', type=int, default=500, 
                       help='Number of images for benchmarking')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device (cuda/cpu)')
    parser.add_argument('--warmup', type=int, default=10, 
                       help='Number of warmup iterations')
    parser.add_argument('--ann_file', type=str, default=None,
                       help='Custom annotation file (e.g., subset file)')
    
    args = parser.parse_args()
    
    # Configuration
    DATA_ROOT = "./data/coco"
    RESULTS_DIR = "./results/benchmark"
    PLOTS_DIR = "./results/plots"
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
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
        split='val',
        year=2017,
        annotations_file=args.ann_file
    )
    
    # Initialize models
    print("\nInitializing models...")
    yolo = YOLODetector(model_variant='yolov8n', device=args.device)
    faster_rcnn = FasterRCNNDetector(backbone='resnet50', device=args.device)
    
    # Create benchmark
    benchmark = SpeedBenchmark(device=args.device)
    
    # Benchmark YOLO
    print("\n" + "="*80)
    print("BENCHMARKING YOLO")
    print("="*80)
    yolo_results = benchmark.benchmark_model(
        yolo, dataset_loader, 
        num_images=args.num_images, 
        warmup=args.warmup
    )
    
    # Benchmark Faster R-CNN
    print("\n" + "="*80)
    print("BENCHMARKING FASTER R-CNN")
    print("="*80)
    faster_rcnn_results = benchmark.benchmark_model(
        faster_rcnn, dataset_loader,
        num_images=args.num_images,
        warmup=args.warmup
    )
    
    # Load Task A metrics if available
    taskA_results_file = "./results/metrics/taskA_results.json"
    if os.path.exists(taskA_results_file):
        print("\nLoading Task A metrics...")
        with open(taskA_results_file, 'r') as f:
            taskA_results = json.load(f)
        
        # Add mAP values to benchmark results
        yolo_results.update(taskA_results['yolo']['metrics'])
        faster_rcnn_results.update(taskA_results['faster_rcnn']['metrics'])
    else:
        print("\n⚠️  Warning: Task A results not found.")
        print("   Run Task A first to get mAP values for the plot.")
        print("   Using placeholder values for demonstration.")
        yolo_results['mAP@[0.5:0.95]'] = 0.37
        faster_rcnn_results['mAP@[0.5:0.95]'] = 0.42
    
    # Combine results
    results = {
        'YOLOv8n': yolo_results,
        'Faster R-CNN (R50)': faster_rcnn_results
    }
    
    # Save results
    output_file = os.path.join(RESULTS_DIR, 'taskB_results.json')
    with open(output_file, 'w') as f:
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in result.items() if k != 'model_info'
            }
        json.dump(json_results, f, indent=4)
    
    print(f"\nResults saved to {output_file}")
    
    # Print comparison
    print("\n" + "="*80)
    print("SPEED-ACCURACY COMPARISON")
    print("="*80)
    print(f"\n{'Model':<25} {'FPS':<12} {'Inference Time':<18} {'mAP@[0.5:0.95]':<15}")
    print("-"*80)
    
    for model_name, result in results.items():
        fps = result['fps']
        time_ms = result['mean_time'] * 1000
        map_val = result.get('mAP@[0.5:0.95]', 0) * 100
        
        print(f"{model_name:<25} {fps:>7.2f} FPS  {time_ms:>10.2f} ms      "
              f"{map_val:>6.2f}%")
    
    print("="*80)
    
    # Calculate speed-up
    speedup = yolo_results['fps'] / faster_rcnn_results['fps']
    accuracy_loss = (yolo_results.get('mAP@[0.5:0.95]', 0) - 
                     faster_rcnn_results.get('mAP@[0.5:0.95]', 0)) * 100
    
    print("\nKEY FINDINGS:")
    print("-" * 80)
    print(f"⚡ YOLO is {speedup:.2f}x faster than Faster R-CNN")
    print(f"📉 YOLO has {abs(accuracy_loss):.2f}% {'lower' if accuracy_loss < 0 else 'higher'} mAP")
    print(f"\n💡 Trade-off Analysis:")
    print(f"   - For every {abs(accuracy_loss):.2f}% drop in accuracy,")
    print(f"     YOLO gains {speedup:.2f}x speed improvement")
    print(f"   - YOLO: Suitable for real-time applications (surveillance, robotics)")
    print(f"   - Faster R-CNN: Better for accuracy-critical tasks (medical imaging)")
    
    # Create visualization
    print("\nCreating speed-accuracy trade-off plot...")
    plot_path = os.path.join(PLOTS_DIR, 'speed_accuracy_tradeoff.png')
    create_speed_accuracy_plot(results, plot_path)
    
    # Also create PDF version
    plot_path_pdf = os.path.join(PLOTS_DIR, 'speed_accuracy_tradeoff.pdf')
    plt.savefig(plot_path_pdf, format='pdf', dpi=300, bbox_inches='tight')
    print(f"PDF version saved to {plot_path_pdf}")
    
    print("\n✓ Task B completed successfully!")
    print("  Next steps:")
    print("  1. Analyze failure cases: python src/visualization/failure_cases.py")
    print("  2. Create comparison viewer: python src/visualization/comparison_viewer.py")
    print("  3. Generate report: Use the data in results/ directory")


if __name__ == "__main__":
    main()
