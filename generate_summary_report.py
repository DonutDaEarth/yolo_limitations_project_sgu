"""
Generate comprehensive summary report for YOLO Limitations Project
"""
import json
import os
from datetime import datetime

def load_json(filepath):
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def generate_markdown_report():
    """Generate markdown summary report"""
    
    # Load results
    taskA_results = load_json('results/metrics/taskA_results.json')
    taskB_results = load_json('results/benchmark/taskB_results.json')
    failure_cases = load_json('results/failure_cases/failure_cases.json')
    
    report = []
    report.append("# YOLO Limitations Analysis - Project Summary Report")
    report.append(f"\n**Date**: {datetime.now().strftime('%B %d, %Y')}")
    report.append(f"\n**Dataset**: COCO 2017 Validation (500 images)")
    report.append("\n---\n")
    
    # Executive Summary
    report.append("## Executive Summary\n")
    report.append("This project quantifies the limitations of single-shot detectors (YOLO) compared to ")
    report.append("two-stage detectors (Faster R-CNN) on the COCO dataset, with particular focus on ")
    report.append("small object detection and speed-accuracy trade-offs.\n")
    
    # Task A Results
    report.append("\n## Task A: Detection Accuracy Comparison\n")
    report.append("### YOLO (YOLOv8n)\n")
    yolo_metrics = taskA_results['yolo']['metrics']
    report.append(f"- **mAP@[0.5:0.95]**: {yolo_metrics['mAP@[0.5:0.95]']*100:.2f}%\n")
    report.append(f"- **mAP@0.5**: {yolo_metrics['mAP@0.5']*100:.2f}%\n")
    report.append(f"- **mAP (Small Objects)**: {yolo_metrics['mAP(Small)']*100:.2f}%\n")
    report.append(f"- **mAP (Medium Objects)**: {yolo_metrics['mAP(Medium)']*100:.2f}%\n")
    report.append(f"- **mAP (Large Objects)**: {yolo_metrics['mAP(Large)']*100:.2f}%\n")
    
    report.append("\n### Faster R-CNN (ResNet-50)\n")
    frcnn_metrics = taskA_results['faster_rcnn']['metrics']
    report.append(f"- **mAP@[0.5:0.95]**: {frcnn_metrics['mAP@[0.5:0.95]']*100:.2f}%\n")
    report.append(f"- **mAP@0.5**: {frcnn_metrics['mAP@0.5']*100:.2f}%\n")
    report.append(f"- **mAP (Small Objects)**: {frcnn_metrics['mAP(Small)']*100:.2f}%\n")
    report.append(f"- **mAP (Medium Objects)**: {frcnn_metrics['mAP(Medium)']*100:.2f}%\n")
    report.append(f"- **mAP (Large Objects)**: {frcnn_metrics['mAP(Large)']*100:.2f}%\n")
    
    # Task B Results
    report.append("\n## Task B: Speed Benchmark\n")
    yolo_speed = taskB_results['YOLOv8n']
    frcnn_speed = taskB_results['Faster R-CNN (R50)']
    
    report.append("### Performance Metrics\n")
    report.append("| Metric | YOLO | Faster R-CNN | Speedup |\n")
    report.append("|--------|------|--------------|----------|\n")
    report.append(f"| **FPS** | {yolo_speed['fps']:.2f} | {frcnn_speed['fps']:.2f} | {yolo_speed['fps']/frcnn_speed['fps']:.1f}x |\n")
    report.append(f"| **Inference Time (ms)** | {yolo_speed['mean_time']*1000:.1f} | {frcnn_speed['mean_time']*1000:.1f} | - |\n")
    report.append(f"| **Avg Detections** | {yolo_speed['avg_detections']:.1f} | {frcnn_speed['avg_detections']:.1f} | - |\n")
    
    # Key Findings
    report.append("\n## Key Findings\n")
    
    # Small object performance
    small_diff = (frcnn_metrics['mAP(Small)'] - yolo_metrics['mAP(Small)']) * 100
    if abs(small_diff) > 0.01:
        if small_diff > 0:
            report.append(f"1. **Small Object Detection**: Faster R-CNN outperforms YOLO on small objects by {small_diff:.2f}%, ")
            report.append("demonstrating YOLO's limitation in detecting fine-grained details.\n\n")
        else:
            report.append(f"1. **Small Object Detection**: YOLO performs comparably to Faster R-CNN on small objects.\n\n")
    
    # Speed advantage
    speedup = yolo_speed['fps'] / frcnn_speed['fps']
    report.append(f"2. **Speed Advantage**: YOLO is **{speedup:.1f}x faster** than Faster R-CNN ")
    report.append(f"({yolo_speed['fps']:.1f} FPS vs {frcnn_speed['fps']:.2f} FPS on CPU).\n\n")
    
    # Trade-offs
    report.append("3. **Speed-Accuracy Trade-off**: YOLO sacrifices some accuracy for real-time performance, ")
    report.append("making it suitable for applications where speed is critical.\n\n")
    
    # Failure cases
    report.append(f"4. **Failure Analysis**: Identified {len(failure_cases)} critical failure cases including:\n")
    fn_cases = [f for f in failure_cases if f.get('type') == 'false_negative']
    pl_cases = [f for f in failure_cases if f.get('type') == 'poor_localization']
    report.append(f"   - {len(fn_cases)} false negative cases (objects missed by YOLO)\n")
    report.append(f"   - {len(pl_cases)} poor localization cases (inaccurate bounding boxes)\n\n")
    
    # Recommendations
    report.append("\n## Recommendations\n")
    report.append("1. **Use YOLO when**:\n")
    report.append("   - Real-time performance is critical (video processing, robotics)\n")
    report.append("   - Computing resources are limited\n")
    report.append("   - Detecting large/medium objects in simple scenes\n\n")
    
    report.append("2. **Use Faster R-CNN when**:\n")
    report.append("   - Accuracy is paramount\n")
    report.append("   - Detecting small objects or dense scenes\n")
    report.append("   - Processing time is not a constraint\n\n")
    
    report.append("3. **Hybrid Approaches**:\n")
    report.append("   - Use YOLO for initial filtering, Faster R-CNN for refinement\n")
    report.append("   - Apply YOLO for real-time tracking, periodic Faster R-CNN validation\n\n")
    
    # Files Generated
    report.append("\n## Generated Outputs\n")
    report.append("### Metrics\n")
    report.append("- `results/metrics/taskA_results.json` - Accuracy metrics\n")
    report.append("- `results/metrics/predictions.npz` - Model predictions\n\n")
    
    report.append("### Benchmark\n")
    report.append("- `results/benchmark/taskB_results.json` - Speed metrics\n")
    report.append("- `results/plots/speed_accuracy_tradeoff.png` - Visualization\n\n")
    
    report.append("### Failure Analysis\n")
    report.append("- `results/failure_cases/failure_cases.json` - Failure case data\n\n")
    
    # Conclusion
    report.append("\n## Conclusion\n")
    report.append("This analysis demonstrates the fundamental trade-off between speed and accuracy in ")
    report.append("object detection. While YOLO provides exceptional speed advantages, it shows limitations ")
    report.append("in certain scenarios, particularly with small objects. The choice between single-shot ")
    report.append("and two-stage detectors should be driven by specific application requirements.\n")
    
    # Write report
    report_path = 'ANALYSIS_SUMMARY.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    print(f"✅ Summary report generated: {report_path}")
    return report_path

if __name__ == "__main__":
    generate_markdown_report()
