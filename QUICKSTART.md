# YOLO Limitations Project - Quick Start Guide

## 🚀 Getting Started (15 Minutes)

This guide will help you set up and run the entire project quickly.

### Step 1: Installation (5 minutes)

```powershell
# Navigate to project directory
cd "c:\Users\ss1ku\01 STEVEN FILES\SGU\7th Semester\Pattern Recognition & Image Processing\computer_vision\yolo_limitations_project"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pycocotools matplotlib seaborn pandas numpy pyyaml tqdm

# For Faster R-CNN (Detectron2) - may need Visual Studio Build Tools
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

**Note:** If Detectron2 installation fails, you can use pre-built wheels or install Visual Studio Build Tools.

### Step 2: Download COCO Dataset (10-30 minutes depending on internet speed)

```powershell
# Create data directory
mkdir data\coco -Force
cd data\coco

# Download COCO 2017 validation images and annotations
# Option 1: Using wget (if available)
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Option 2: Manual download
# Go to: https://cocodataset.org/#download
# Download: val2017.zip and annotations_trainval2017.zip

# Extract
Expand-Archive val2017.zip -DestinationPath .
Expand-Archive annotations_trainval2017.zip -DestinationPath .

cd ..\..
```

**Alternative (Faster):** Use a subset for testing

```powershell
# Download only 500 images (for quick testing)
python scripts/download_coco_subset.py --num_images 500
```

### Step 3: Run Experiments (20-30 minutes on GPU)

```powershell
# Task A: Small Object & Clutter Challenge
python scripts/run_taskA.py --num_images 500 --small_objects_only

# Task B: Speed Benchmark
python scripts/run_taskB.py --num_images 500

# Analyze Failure Cases
python src/visualization/failure_cases.py

# Generate Comparison Visualizations
python src/visualization/comparison_viewer.py --generate --num_images 20
```

## 📊 Expected Outputs

After running all scripts, you should have:

```
results/
├── metrics/
│   ├── taskA_results.json          # mAP metrics for both models
│   └── predictions.npz             # Raw predictions for analysis
├── benchmark/
│   └── taskB_results.json          # Speed benchmark results
├── plots/
│   ├── speed_accuracy_tradeoff.png # Main visualization for report
│   └── speed_accuracy_tradeoff.pdf # PDF version
├── failure_cases/
│   ├── failure_case_01.png         # Top 10 failure cases
│   ├── ...
│   └── failure_case_10.png
└── comparisons/
    ├── comparison_001.png          # Side-by-side comparisons
    ├── ...
    └── comparison_020.png
```

## 🎯 Quick Test (Without COCO Download)

If you want to test the code without downloading COCO:

```powershell
# Test model loading
python -c "from src.models.yolo_detector import YOLODetector; detector = YOLODetector(); print('✓ YOLO OK')"

python -c "from src.models.two_stage_detector import FasterRCNNDetector; detector = FasterRCNNDetector(); print('✓ Faster R-CNN OK')"
```

## 🔧 Troubleshooting

### CUDA Out of Memory

```powershell
# Use smaller batch size or CPU
python scripts/run_taskA.py --device cpu
```

### Detectron2 Installation Issues

```powershell
# Use pre-built wheel for Windows
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html

# Or build from source (requires Visual Studio)
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

### Missing COCO Dataset

```powershell
# Verify dataset structure
ls data\coco\val2017  # Should show images
ls data\coco\annotations  # Should show instances_val2017.json
```

## 📝 For Your Report

### Key Numbers to Extract

After running experiments, extract these values for your report:

```powershell
# View Task A results
cat results\metrics\taskA_results.json

# View Task B results
cat results\benchmark\taskB_results.json
```

**Fill in your report template with:**

- YOLO mAP@[0.5:0.95]: `results['yolo']['metrics']['mAP@[0.5:0.95]']`
- Faster R-CNN mAP@[0.5:0.95]: `results['faster_rcnn']['metrics']['mAP@[0.5:0.95]']`
- YOLO FPS: `results['YOLOv8n']['fps']`
- Faster R-CNN FPS: `results['Faster R-CNN (R50)']['fps']`
- Speed-up factor: `YOLO_FPS / FRCNN_FPS`

### Figures for Report

1. **Figure 1 (Speed-Accuracy):** `results/plots/speed_accuracy_tradeoff.png`
2. **Figures 2-11 (Failures):** `results/failure_cases/failure_case_*.png`
3. **Comparison Examples:** `results/comparisons/comparison_*.png`

## 🎬 For Live Demo

### Demo 1: Speed Benchmark (3 minutes)

```powershell
python scripts/run_taskB.py --num_images 50
```

Shows real-time FPS calculation for both models.

### Demo 2: Side-by-Side Comparison (5 minutes)

```powershell
python src/visualization/comparison_viewer.py --interactive
```

Navigate through images with:

- `n` = Next image
- `p` = Previous image
- `s` = Save current comparison
- `q` = Quit

## ⏱️ Timeline Recommendation

**Week 1:**

- Day 1-2: Setup environment, download dataset
- Day 3-4: Run Task A experiments
- Day 5: Run Task B experiments

**Week 2:**

- Day 1-2: Analyze failure cases
- Day 3-4: Generate visualizations
- Day 5: Start writing report

**Week 3:**

- Day 1-3: Complete report
- Day 4: Prepare presentation
- Day 5: Practice demo

## 💡 Tips for Success

1. **Run experiments early** - Don't wait until last week
2. **Save all outputs** - You'll need them for report and presentation
3. **Document issues** - Any problems you encounter are interesting discussion points
4. **Test demos** - Practice the live demonstration multiple times
5. **Backup results** - Save to multiple locations (cloud, USB)

## 📧 Need Help?

- Check `README.md` for detailed documentation
- Review code comments in each Python file
- Test individual components before running full pipeline
- Use smaller datasets for debugging

---

**Good luck with your project! 🎓**
