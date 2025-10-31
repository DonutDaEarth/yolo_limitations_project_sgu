# Quantifying the Limitations of Single-Shot Detectors

**Course:** Pattern Recognition & Image Processing  
**Project Type:** Midterm Project  
**Duration:** 3 Weeks  
**Team Size:** 2-3 Members

## Project Overview

This project aims to quantitatively and qualitatively demonstrate the trade-offs and specific failure modes of high-efficiency, single-stage object detection models (YOLO) compared to two-stage detectors (Faster R-CNN).

### Target Statement for Limitation

_"Single-shot object detectors like YOLO prioritize efficiency and real-time performance while maintaining high accuracy, making them suitable for many practical applications."_

## Project Structure

```
yolo_limitations_project/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml                 # Configuration for models and datasets
├── src/
│   ├── models/
│   │   ├── yolo_detector.py       # YOLO model wrapper
│   │   └── two_stage_detector.py  # Faster R-CNN wrapper
│   ├── data/
│   │   ├── dataset_loader.py      # Dataset loading utilities
│   │   └── preprocessing.py       # Data preprocessing
│   ├── evaluation/
│   │   ├── metrics.py             # mAP, mAP(Small) calculations
│   │   ├── inference.py           # Run inference on datasets
│   │   └── speed_benchmark.py     # FPS measurement (Task B)
│   ├── visualization/
│   │   ├── failure_cases.py       # Identify failure modes
│   │   └── comparison_viewer.py   # Side-by-side visualization tool
│   └── utils/
│       └── helpers.py             # Utility functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_results_analysis.ipynb
├── scripts/
│   ├── run_taskA.py               # Run Task A experiments
│   ├── run_taskB.py               # Run Task B experiments
│   └── generate_report_data.py    # Generate all results for report
├── results/
│   ├── metrics/                   # JSON/CSV files with metrics
│   ├── plots/                     # Speed-accuracy trade-off plots
│   └── failure_cases/             # Images showing failure modes
├── report/
│   ├── technical_report.md        # Main report (markdown)
│   └── figures/                   # Figures for report
└── presentation/
    └── slides.pptx                # Presentation slides
```

## Quick Start

### 1. Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup

Download COCO 2017 validation dataset:

```powershell
# Download annotations and images (script will guide you)
python src/data/dataset_loader.py --download
```

### 3. Run Experiments

**Task A: Small Object & Clutter Challenge**

```powershell
python scripts/run_taskA.py --dataset coco --split val
```

**Task B: Speed vs. Accuracy Trade-off**

```powershell
python scripts/run_taskB.py --num_images 500
```

### 4. Generate Visualizations

```powershell
# Launch interactive comparison viewer
python src/visualization/comparison_viewer.py

# Generate failure case report
python src/visualization/failure_cases.py --top_k 10
```

## Key Deliverables

### 1. Technical Report (5-7 Pages)

- Introduction with model and dataset selection
- Methodology with hardware/software setup
- Quantitative results (mAP, mAP(Small), FPS)
- Discussion of YOLO limitations
- Conclusion

### 2. Live Presentation (15 Minutes)

- Focus on limitations and failures (8+ minutes)
- Speed-accuracy trade-off visualization
- Clear, professional presentation

### 3. Code Demonstration (15 Minutes)

- **Demo 1:** Live FPS and mAP benchmark
- **Demo 2:** Side-by-side failure mode visualization

## Models Used

- **Single-Stage Detector:** YOLOv8 (latest stable)
- **Two-Stage Detector:** Faster R-CNN (ResNet-50 backbone)

## Datasets

- **Primary:** COCO 2017 (subset focusing on small objects)
- **Optional:** SKU-110K, ExDARK (for additional challenges)

## Evaluation Metrics

1. **mAP@[0.5:0.95]** - Standard object detection performance
2. **mAP@0.5** - IoU threshold of 0.5
3. **mAP(Small)** - Mean Average Precision for objects with Area < 32² pixels
4. **FPS** - Frames Per Second (inference speed)
5. **Localization Error** - IoU analysis for dense clusters

## Team Members

- Member 1: [Name]
- Member 2: [Name]
- Member 3: [Name] (Optional)

## Timeline

| Week   | Tasks                                              |
| ------ | -------------------------------------------------- |
| Week 1 | Setup, dataset preparation, model implementation   |
| Week 2 | Run experiments (Task A & B), collect metrics      |
| Week 3 | Analysis, report writing, presentation prep, demos |

## Hardware Requirements

- **Recommended:** NVIDIA GPU (8GB+ VRAM)
- **Minimum:** CPU with 16GB RAM (slower inference)

## References

1. Redmon, J., et al. "You Only Look Once: Unified, Real-Time Object Detection"
2. Ren, S., et al. "Faster R-CNN: Towards Real-Time Object Detection"
3. Lin, T., et al. "Microsoft COCO: Common Objects in Context"

## License

Academic Use Only - For educational purposes in Pattern Recognition course.
