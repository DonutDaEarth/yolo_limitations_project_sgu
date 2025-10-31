# Quick Command Reference - Code Demo

## 🚀 ESSENTIAL COMMANDS FOR DEMO

### 1. Navigate to Project

```powershell
cd "C:\Users\ss1ku\01 STEVEN FILES\SGU\7th Semester\Pattern Recognition & Image Processing\computer_vision\yolo_limitations_project"
```

### 2. Activate Environment

```powershell
..\venv\Scripts\Activate.ps1
```

### 3. Test Models

```powershell
# YOLO test
..\venv\Scripts\python.exe test_yolo.py

# Faster R-CNN test
..\venv\Scripts\python.exe test_faster_rcnn.py
```

### 4. Show Results

```powershell
# Task A results (accuracy)
Get-Content results\metrics\taskA_results.json

# Task B results (speed)
Get-Content results\benchmark\taskB_results.json

# Failure analysis
Get-Content results\failure_cases\failure_cases.json | Select-Object -First 30
```

### 5. Show Visualizations

```powershell
# Speed-accuracy plot
Start-Process results\plots\speed_accuracy_tradeoff.png

# Summary document
code ANALYSIS_SUMMARY.md
```

---

## 📊 KEY NUMBERS TO MEMORIZE

- **44.1x faster**: YOLO speedup (20.45 FPS vs 0.46 FPS)
- **0.0% vs 0.033%**: Small object mAP (YOLO vs Faster R-CNN)
- **~49ms vs ~2156ms**: Inference time per image
- **5.2 vs 12.6**: Average detections per image
- **954**: False negative cases
- **1,245**: Poor localization cases
- **68.2%**: Failures due to small object size

---

## 🎯 DEMO FLOW (15 MIN)

```
[0-2 min]   Setup: Navigate, activate venv, show structure
[2-5 min]   Models: Test YOLO + Faster R-CNN loading
[5-10 min]  Results: Task A accuracy + Task B speed
[10-13 min] Analysis: Failure cases + summary
[13-15 min] Wrap-up: Conclusions + Q&A
```

---

## 🎤 KEY TALKING POINTS

### When showing YOLO test:

- "6.2MB model, loads instantly"
- "YOLOv8n - nano variant for speed"

### When showing Faster R-CNN test:

- "167MB model with ResNet-50"
- "Two-stage architecture"

### When showing Task A results:

- "YOLO: 0.0% on small objects - completely missed them"
- "Faster R-CNN: 0.033% - detects small objects"
- "This is the fundamental limitation"

### When showing Task B results:

- "20 FPS vs 0.46 FPS = 44x speedup"
- "49ms vs 2156ms per image"
- "Real-time capability vs batch processing"

### When showing failure analysis:

- "954 false negatives - objects YOLO missed"
- "1,245 poor localizations - inaccurate boxes"
- "68% due to small size - proves our hypothesis"

---

## ⚡ BACKUP COMMANDS (if needed)

### Show installed packages:

```powershell
pip list | Select-String -Pattern "torch|ultralytics|detectron2"
```

### Check dataset:

```powershell
dir data\coco\val2017 | Measure-Object
# Should show 500 items
```

### Check all results exist:

```powershell
dir results -Recurse -File
```

### Quick inference demo:

```powershell
..\venv\Scripts\python.exe -c "from src.models.yolo_detector import YOLODetector; import time; d = YOLODetector(device='cpu'); import cv2; img = cv2.imread('data/coco/val2017/000000000139.jpg'); s = time.time(); r = d.predict(img); print(f'Detected {len(r[\"boxes\"])} objects in {(time.time()-s)*1000:.1f}ms')"
```

---

## 🎓 QUESTIONS & ANSWERS

**Q: Why CPU instead of GPU?**
→ "To simulate resource-constrained deployment. GPU would be faster but same 40-50x relative speedup."

**Q: Can YOLO improve on small objects?**
→ "Larger YOLO variants (m/l/x) improve somewhat, but grid architecture fundamentally limits small object detection."

**Q: Real-world applications?**
→ "YOLO: traffic monitoring, sports, robotics. Faster R-CNN: medical imaging, satellite, dense crowds."

**Q: How to choose?**
→ "If real-time required and objects are medium/large: YOLO. If accuracy critical and small objects: Faster R-CNN."

---

## 📱 DEMO DAY CHECKLIST

- [ ] Laptop charged
- [ ] Project folder open
- [ ] Terminal ready
- [ ] Results files verified to exist
- [ ] Practiced timing (aim for 12-13 min, leave 2-3 for Q&A)
- [ ] Memorized key numbers
- [ ] Know how to pronounce "YOLO" and "Faster R-CNN"
- [ ] Confident and ready! 💪

---

## 🎬 THE ACTUAL COMMANDS YOU'LL TYPE

**Demo Script (copy-paste ready):**

```powershell
# 1. Navigate
cd "C:\Users\ss1ku\01 STEVEN FILES\SGU\7th Semester\Pattern Recognition & Image Processing\computer_vision\yolo_limitations_project"

# 2. Show structure
dir

# 3. Test YOLO
..\venv\Scripts\python.exe test_yolo.py

# 4. Test Faster R-CNN
..\venv\Scripts\python.exe test_faster_rcnn.py

# 5. Show Task A results
Get-Content results\metrics\taskA_results.json

# 6. Show Task B results
Get-Content results\benchmark\taskB_results.json

# 7. Show plot
Start-Process results\plots\speed_accuracy_tradeoff.png

# 8. Show summary
code ANALYSIS_SUMMARY.md

# Done! Ready for Q&A
```

**Total time: ~12 minutes + 3 min Q&A = 15 minutes** ✅
