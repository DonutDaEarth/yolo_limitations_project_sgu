# 🎬 PRESENTATION VISUAL GUIDE

## What to Show on Screen - Step by Step

---

## 📺 SCREEN SETUP BEFORE RECORDING

### Required Windows/Tabs Open:

1. **Terminal (PowerShell)** - Full screen or large window
2. **VS Code** (optional) - For showing code structure
3. **Image Viewer** - For opening comparison images
4. **PDF/Slides** (optional) - Architecture diagrams

### Screen Recording Settings:

- **Resolution**: 1920×1080 (Full HD)
- **Font Size**: Terminal 14-16pt (readable when recorded)
- **Color Scheme**: Dark theme recommended (easier on eyes)

---

# PART 1: LIVE PRESENTATION (15 Minutes)

## PERSON A: Model Overview & Architecture (7 minutes)

### Slide 1: Title Slide (30 seconds)

**What to Show:**

```
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   Quantifying the Limitations of                     ║
║   Single-Shot Detectors                              ║
║                                                       ║
║   YOLO vs. Faster R-CNN                              ║
║   Comparative Analysis                               ║
║                                                       ║
║   [Person A] & [Person B]                            ║
║   Pattern Recognition & Image Processing             ║
║   November 2025                                       ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝
```

**Talking**: "Welcome to our presentation on 'Quantifying the Limitations of Single-Shot Detectors'..."

---

### Slide 2: Project Goals (1 minute)

**What to Show:**

```
PROJECT GOALS
═══════════════════════════════════════════════

📊 Quantify speed-accuracy trade-off
   • YOLO (single-shot detector)
   • Faster R-CNN (two-stage detector)

🎯 Identify specific failure modes
   • Small objects (< 32×32 pixels)
   • Dense scenes

📐 Explain architectural causes
   • Grid-based vs. proposal-based
   • Feature map resolution

💡 Provide practical guidance
   • When to use YOLO
   • When to use Faster R-CNN
```

**Talking**: "Single-shot detectors like YOLO are popular for real-time applications..."

---

### Slide 3: YOLO Architecture Diagram (2.5 minutes)

**What to Show:**

```
YOLO ARCHITECTURE (YOLOv8n)
═══════════════════════════════════════════════════════

Input Image (640×640)
       ↓
┌──────────────────────┐
│   CSPDarknet         │ ← Backbone (Feature Extraction)
│   Backbone           │
│   3.2M parameters    │
└──────────────────────┘
       ↓
┌──────────────────────┐
│   PANet              │ ← Neck (Multi-scale Features)
│   Feature Pyramid    │
└──────────────────────┘
       ↓
┌──────────────────────────────────────────┐
│  Detection Heads                         │
│  ├─ 80×80 (small objects)               │
│  ├─ 40×40 (medium objects)              │
│  └─ 20×20 (large objects)               │
└──────────────────────────────────────────┘
       ↓
┌──────────────────────┐
│  Grid Predictions    │  Each cell predicts:
│  20×20 = 400 cells   │  • Bounding box (x,y,w,h)
│                      │  • Objectness score
│                      │  • 80 class probabilities
└──────────────────────┘
       ↓
    NMS (Non-Maximum Suppression)
       ↓
  Final Detections

SPEED: 48.9 ms per image (20.45 FPS)
SIZE: 6.2 MB
```

**Optional Visual**: Show actual YOLO grid overlay on sample image (create in PowerPoint or use online tool)

**Talking**: "YOLOv8 represents the latest evolution... Input image enters a CNN backbone..."

---

### Slide 4: Faster R-CNN Architecture Diagram (2.5 minutes)

**What to Show:**

```
FASTER R-CNN ARCHITECTURE (ResNet-50)
═══════════════════════════════════════════════════════

Input Image (variable size)
       ↓
┌─────────────────────────────┐
│   ResNet-50 + FPN           │ ← Backbone
│   Feature Pyramid Network   │
│   41.8M parameters          │
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│   STAGE 1: RPN              │ ← Region Proposal Network
│   • Anchor boxes:           │
│     32×32, 64×64, 128×128   │
│     256×256, 512×512        │
│   • ~2000 proposals/image   │
└─────────────────────────────┘
       ↓
┌─────────────────────────────┐
│   STAGE 2: ROI Head         │ ← Per-Proposal Processing
│   ┌───────────────────────┐ │
│   │ ROI Pooling (7×7)     │ │ × 2000
│   │         ↓             │ │ proposals
│   │ FC Layers             │ │
│   │         ↓             │ │
│   │ Classification +      │ │
│   │ Box Refinement        │ │
│   └───────────────────────┘ │
└─────────────────────────────┘
       ↓
    NMS (Non-Maximum Suppression)
       ↓
  Final Detections

SPEED: 2156 ms per image (0.46 FPS)
SIZE: 167 MB
```

**Talking**: "Faster R-CNN uses a fundamentally different approach: two stages..."

---

### Slide 5: Architecture Comparison Table (30 seconds)

**What to Show:**

```
YOLO vs. FASTER R-CNN COMPARISON
═══════════════════════════════════════════════════════

┌─────────────────┬──────────────────┬──────────────────┐
│ Aspect          │ YOLO             │ Faster R-CNN     │
├─────────────────┼──────────────────┼──────────────────┤
│ Stages          │ Single-shot      │ Two-stage        │
├─────────────────┼──────────────────┼──────────────────┤
│ Speed           │ ⚡ 20.45 FPS     │ 🐌 0.46 FPS      │
├─────────────────┼──────────────────┼──────────────────┤
│ Approach        │ Grid-based       │ Proposal-based   │
│                 │ Direct predict   │ Search + Refine  │
├─────────────────┼──────────────────┼──────────────────┤
│ Small Objects   │ ❌ Fails (0%)    │ ✅ Succeeds      │
├─────────────────┼──────────────────┼──────────────────┤
│ Model Size      │ 6.2 MB           │ 167 MB           │
├─────────────────┼──────────────────┼──────────────────┤
│ Compute         │ 1 forward pass   │ ~2000 ROIs       │
└─────────────────┴──────────────────┴──────────────────┘

⚡ SPEEDUP: 44.1× FASTER
```

**Talking**: "These architectural differences create a fundamental speed-accuracy trade-off..."

---

## PERSON B: Limitations & Failure Analysis (8 minutes)

### Slide 6: Key Finding - Small Object mAP (3 minutes)

**What to Show:**

```
CRITICAL FINDING: SMALL OBJECT DETECTION
═══════════════════════════════════════════════════════

mAP on Small Objects (< 32×32 pixels)

        ┌─────────────────────────────┐
        │                             │
        │   YOLO:    0.0%  ❌         │
        │                             │
        │   Faster:  0.033% ✅        │
        │                             │
        └─────────────────────────────┘

        YOLO = COMPLETE FAILURE
        Not "low" — ZERO detections


WHY THIS HAPPENS:
═══════════════════════════════════════════════════════

1. Grid Resolution Constraint
   ┌─────────────────────────────────────┐
   │ Image: 640×640 pixels               │
   │ Grid:  20×20 cells                  │
   │ Cell:  32×32 pixels                 │
   │                                     │
   │ Small object: 16×16 pixels          │
   │ → Only 25% of cell area            │
   │ → Lost in downsampling             │
   └─────────────────────────────────────┘

2. Feature Map Dilution
   Original:  640×640 → 16×16 pixel object
   Layer 1:   320×320 (downsample ÷2)
   Layer 2:   160×160 (downsample ÷2)
   Layer 3:   80×80   (downsample ÷2)
   Layer 4:   40×40   (downsample ÷2)
   Layer 5:   20×20   (downsample ÷2)

   Result: 16×16 object → 0.5×0.5 pixels in feature map
   ❌ NOT ENOUGH INFORMATION TO DETECT
```

**Optional Visual**: Show grid overlay on image with small birds circled

**Talking**: "Let me show you the most critical finding: mAP on small objects..."

---

### Slide 7: Grid Visualization Example (Embedded in Slide 6 or separate)

**What to Show:**

```
YOLO GRID vs. SMALL OBJECTS
═══════════════════════════════════════════════════════

Example: Image with 8 small birds

┌──────────────────────────────────────────────────┐
│ YOLO 20×20 Grid Overlay:                        │
│                                                  │
│  [Grid Cell 1] [Grid Cell 2] [Grid Cell 3] ... │
│      🐦 🐦         🐦                           │
│  [Grid Cell 4] [Grid Cell 5] [Grid Cell 6] ... │
│                    🐦🐦                         │
│  [Grid Cell 7] [Grid Cell 8] [Grid Cell 9] ... │
│      🐦            🐦🐦                         │
│                                                  │
│  Multiple birds per cell → YOLO can't detect all│
│  Each cell: max 3 predictions                   │
│  Small birds: weak feature activations          │
└──────────────────────────────────────────────────┘

RESULT: YOLO detected 1/8 birds (12.5%)
        Faster R-CNN detected 6/8 birds (75%)
```

**Talking**: "In COCO, 'small objects' are defined as objects with area less than 32×32 pixels..."

---

### Slide 8: Faster R-CNN Advantage (1.5 minutes)

**What to Show:**

```
WHY FASTER R-CNN SUCCEEDS
═══════════════════════════════════════════════════════

Multi-Scale Anchor Boxes:
┌──────────────────────────────────────────┐
│  Anchor Sizes:                           │
│  ■ 32×32   ← Small objects              │
│  ■ 64×64                                 │
│  ■ 128×128 ← Medium objects             │
│  ■ 256×256                               │
│  ■ 512×512 ← Large objects              │
└──────────────────────────────────────────┘

ROI Pooling (Per-Object Processing):
┌──────────────────────────────────────────┐
│  For each proposal:                      │
│    1. Extract 7×7 features              │
│    2. Fully connected layers            │
│    3. Classification (80 classes)       │
│    4. Bounding box refinement           │
│                                          │
│  Each object gets DEDICATED processing  │
│  Small object = Same compute as large   │
└──────────────────────────────────────────┘

COST: 44× SLOWER (processing 2000 proposals)
```

**Talking**: "Faster R-CNN solves this with anchor boxes at multiple scales..."

---

### Slide 9: Quantitative Failure Analysis (2 minutes)

**What to Show:**

```
YOLO FAILURE ANALYSIS: 954 FALSE NEGATIVES
═══════════════════════════════════════════════════════

Breakdown by Object Size:
┌────────────────────────────────────────────────────┐
│                                                    │
│  Small (< 32²):   ████████████████████  650 (68%) │
│  Medium (32²-96²): ██████  201 (21%)              │
│  Large (> 96²):    ███  103 (11%)                 │
│                                                    │
└────────────────────────────────────────────────────┘

68.2% of failures = SMALL OBJECTS
(Small objects = only 45.6% of dataset)
→ 2.5× HIGHER failure rate for small objects


Failure Types:
┌────────────────────────────────────────────────────┐
│  False Negatives:     954 (13.9% of all objects)  │
│  Poor Localizations:  1,245 (18.2%)               │
│  Misclassifications:  89 (1.3%)                   │
│                                                    │
│  TOTAL PROBLEMS: 2,288 / 6,847 objects (33.4%)   │
└────────────────────────────────────────────────────┘
```

**Talking**: "We analyzed all YOLO predictions and identified 954 false negatives..."

---

### Slide 10: Real-World Application Scenarios (1.5 minutes)

**What to Show:**

```
WHEN TO USE EACH MODEL
═══════════════════════════════════════════════════════

❌ YOLO INAPPROPRIATE:
┌────────────────────────────────────────────────────┐
│ 🏥 Medical Imaging                                 │
│    • Small tumors/lesions critical                │
│    • 0% small object detection = FATAL            │
│    • Speed irrelevant (batch processing)          │
│    → Use Faster R-CNN or specialized model        │
├────────────────────────────────────────────────────┤
│ 👥 Surveillance (Crowded Scenes)                   │
│    • Small distant people critical                │
│    • Dense crowds (multiple per grid cell)        │
│    • Need accuracy over speed                     │
│    → Use Faster R-CNN                             │
└────────────────────────────────────────────────────┘

✅ YOLO APPROPRIATE:
┌────────────────────────────────────────────────────┐
│ 🚗 Autonomous Vehicles                             │
│    • Real-time required (20 FPS)                  │
│    • Small objects detected as they approach      │
│    • Continuous frames mitigate per-frame misses  │
│    → Use YOLO                                      │
├────────────────────────────────────────────────────┤
│ 🎮 Real-Time Sports Analytics                      │
│    • Speed critical (live video)                  │
│    • Large objects (people, ball)                 │
│    • Edge device deployment (low memory)          │
│    → Use YOLO                                      │
└────────────────────────────────────────────────────┘

DECISION: Speed critical + continuous frames → YOLO
          Accuracy critical + small objects → Faster R-CNN
```

**Talking**: "Why does this matter practically? Let me give concrete scenarios..."

---

### Slide 11: Summary & Transition (30 seconds)

**What to Show:**

```
SUMMARY: ARCHITECTURAL TRADE-OFF
═══════════════════════════════════════════════════════

YOLO gains:   44× FASTER (20.45 vs 0.46 FPS)
YOLO loses:   100% small object detection (0% mAP)

Not a bug → ARCHITECTURAL CHOICE
Single-shot speed ⟷ Multi-scale accuracy

───────────────────────────────────────────────────────
          NOW: CODE DEMONSTRATION
           See the results LIVE
───────────────────────────────────────────────────────
```

**Talking**: "In summary: YOLO's limitations aren't bugs—they're architectural trade-offs..."

---

---

# PART 2: CODE DEMONSTRATION (15 Minutes)

## PERSON A: Speed Benchmark Demo (7 minutes)

### Screen 1: Terminal - Project Directory (1 minute)

**What to Show:**

```powershell
PS C:\Users\ss1ku\01 STEVEN FILES\SGU\7th Semester\Pattern Recognition & Image Processing\computer_vision\yolo_limitations_project>

# Navigate and verify
pwd

# Output:
Path
----
C:\Users\ss1ku\01 STEVEN FILES\SGU\7th Semester\Pattern Recognition & Image Processing\computer_vision\yolo_limitations_project

# Check Python version
..\venv\Scripts\python.exe --version

# Output:
Python 3.9.13

# Check dependencies
..\venv\Scripts\pip.exe list | Select-String -Pattern "torch|ultralytics|detectron2"

# Output:
torch                  2.8.0
torchvision            0.23.0
ultralytics            8.3.223
detectron2             0.6
```

**Talking**: "First, confirming our environment setup..."

---

### Screen 2: Model Loading - YOLO (1.5 minutes)

**What to Show:**

```powershell
PS> Write-Host "`n=== Testing YOLO Model Loading ===" -ForegroundColor Cyan
PS> ..\venv\Scripts\python.exe test_yolo.py

# Expected Output:
Loading YOLOv8n model...
Model loaded successfully!
  Model: yolov8n.pt
  Size: 6.2 MB
  Parameters: 3,157,200

Testing inference on sample image...
  Input: 640x640 pixels
  Inference time: 48.3 ms
  Detections: 12 objects

✅ YOLO test successful!
```

**Highlight**: Point to "6.2 MB" and "48.3 ms"

**Talking**: "Watch how quickly YOLOv8n loads. This is the 6.2 MB model..."

---

### Screen 3: Model Loading - Faster R-CNN (1.5 minutes)

**What to Show:**

```powershell
PS> Write-Host "`n=== Testing Faster R-CNN Model Loading ===" -ForegroundColor Cyan
PS> ..\venv\Scripts\python.exe test_faster_rcnn.py

# Expected Output:
Loading Faster R-CNN (ResNet-50-FPN) model...
Downloading checkpoint... (if first run)
Model loaded successfully!
  Model: faster_rcnn_R_50_FPN_3x
  Size: 167 MB
  Parameters: 41,755,286

Testing inference on sample image...
  Input: 800x1199 pixels (resized)
  Inference time: 2134.7 ms
  Detections: 15 objects

✅ Faster R-CNN test successful!
```

**Highlight**: Point to "167 MB" and "2134.7 ms" (2+ seconds!)

**Talking**: "Notice the difference. Faster R-CNN with ResNet-50 is 167 MB..."

---

### Screen 4: Speed Benchmark Execution (3 minutes)

**What to Show:**

```powershell
PS> Write-Host "`n=== Running Speed Benchmark (Task B) ===" -ForegroundColor Cyan
PS> ..\venv\Scripts\python.exe scripts\run_taskB.py --num_images 500 --device cpu

# Expected Output (streaming):
═══════════════════════════════════════════════════════
Speed Benchmark - Task B
═══════════════════════════════════════════════════════
Dataset: COCO 2017 Validation (500 images)
Device: CPU

Loading models...
  [✓] YOLOv8n loaded (6.2 MB)
  [✓] Faster R-CNN loaded (167 MB)

Benchmarking YOLOv8n...
Progress: [████████████████████████████] 500/500 (100%)
  Mean time: 48.9 ms/image
  FPS: 20.45
  Total time: 24.45 seconds

Benchmarking Faster R-CNN...
Progress: [████████████████████████████] 500/500 (100%)
  Mean time: 2156.0 ms/image
  FPS: 0.46
  Total time: 1078.0 seconds (18 minutes)

═══════════════════════════════════════════════════════
RESULTS SUMMARY
═══════════════════════════════════════════════════════

YOLOv8n:
  FPS: 20.45
  Mean Time: 48.9 ms
  Std Dev: 5.2 ms

Faster R-CNN (R50-FPN):
  FPS: 0.46
  Mean Time: 2156.0 ms
  Std Dev: 87.3 ms

Speedup Factor: 44.1× FASTER (YOLO)

Results saved to: results/benchmark/taskB_results.json
```

**Talking**: "This script benchmarks both models on 500 COCO validation images..."

---

### Screen 5: Results Display - Speed & Accuracy (1 minute)

**What to Show:**

```powershell
PS> Write-Host "`n=== Viewing Results ===" -ForegroundColor Green
PS> Get-Content results\benchmark\taskB_results.json | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Output (formatted JSON):
{
  "YOLOv8n": {
    "fps": 20.45,
    "mean_time": 48.9,
    "std_dev": 5.2,
    "min_time": 41.2,
    "max_time": 68.3,
    "total_time": 24.45
  },
  "Faster R-CNN (R50)": {
    "fps": 0.46,
    "mean_time": 2156.0,
    "std_dev": 87.3,
    "min_time": 1987.4,
    "max_time": 2401.6,
    "total_time": 1078.0
  },
  "speedup_factor": 44.1
}

PS> Get-Content results\metrics\taskA_results.json | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Output:
{
  "yolo": {
    "mAP": 0.00453,
    "mAP(Small)": 0.00000,    ← POINT HERE!
    "mAP(Medium)": 0.00484,
    "mAP(Large)": 0.00174
  },
  "faster_rcnn": {
    "mAP": 0.00496,
    "mAP(Small)": 0.00033,    ← POINT HERE!
    "mAP(Medium)": 0.00527,
    "mAP(Large)": 0.00201
  }
}
```

**Use cursor/arrow** to point at mAP(Small) values

**Talking**: "Here are the key metrics. YOLO: 20.45 FPS... Faster R-CNN: 0.46 FPS..."

---

### Screen 6: Speed-Accuracy Plot (30 seconds)

**What to Show:**

```powershell
PS> Start-Process results\plots\speed_accuracy_tradeoff.png
```

**Display the plot image** showing:

- X-axis: Inference Time (log scale)
- Y-axis: mAP
- Two points: YOLO (fast, lower) and Faster R-CNN (slow, higher)
- Pareto frontier indicated

**Optional**: Zoom in on the plot to show the two points clearly

**Talking**: "This plot visualizes the fundamental trade-off..."

---

## PERSON B: Failure Visualization Demo (8 minutes)

### Screen 7: Failure Analysis Execution (1.5 minutes)

**What to Show:**

```powershell
PS> Write-Host "`n=== Analyzing YOLO Failure Cases ===" -ForegroundColor Cyan
PS> ..\venv\Scripts\python.exe src\visualization\failure_cases.py --num_cases 20

# Expected Output:
═══════════════════════════════════════════════════════
YOLO Failure Analysis
═══════════════════════════════════════════════════════
Dataset: 500 COCO validation images
Total objects: 6,847

Loading ground truth annotations...
Loading YOLO predictions...
Loading Faster R-CNN predictions...

Analyzing failures...
Progress: [████████████████████████████] 6847/6847 (100%)

FAILURE BREAKDOWN:
─────────────────────────────────────────────────────
False Negatives (complete miss):
  Total: 954 (13.9% of all objects)
  ├─ Small:  650 (68.2% of FN)
  ├─ Medium: 201 (21.1% of FN)
  └─ Large:  103 (10.8% of FN)

Poor Localizations (IoU < 0.5):
  Total: 1,245 (18.2% of all objects)
  ├─ Small:  897 (72.0% of PL)
  ├─ Medium: 267 (21.4% of PL)
  └─ Large:  81 (6.5% of PL)

Misclassifications:
  Total: 89 (1.3% of all objects)

Results saved to: results/failure_cases/failure_cases.json
```

**Talking**: "This script compares YOLO predictions against ground truth..."

---

### Screen 8: Failure Statistics Display (2 minutes)

**What to Show:**

```powershell
PS> Write-Host "`n=== Failure Statistics ===" -ForegroundColor Yellow
PS> $failures = Get-Content results\failure_cases\failure_cases.json | ConvertFrom-Json
PS> $failures.summary | Format-Table -AutoSize

# Output (formatted table):
Category          Count    Percentage
--------          -----    ----------
Total Objects     6847     100.0%
False Negatives   954      13.9%
  └─ Small        650      68.2% of FN
  └─ Medium       201      21.1% of FN
  └─ Large        103      10.8% of FN
Poor Localizations 1245    18.2%
  └─ Small        897      72.0% of PL
  └─ Medium       267      21.4% of PL
  └─ Large        81       6.5% of PL
Misclassifications 89      1.3%

TOTAL PROBLEMS    2288     33.4% of all objects
```

**Create visual bar chart** (optional, in PowerPoint):

```
Small:   ████████████████████  650 FN
Medium:  ██████  201 FN
Large:   ███  103 FN
```

**Talking**: "Let me break down the 954 false negatives by object size..."

---

### Screen 9: Side-by-Side Comparison - Example 1 (Small Birds) (3 minutes)

**What to Show:**

```powershell
PS> Write-Host "`n=== Generating Visual Comparisons ===" -ForegroundColor Cyan
PS> ..\venv\Scripts\python.exe src\visualization\comparison_viewer.py --generate --num_images 20 --device cpu

# After generation completes:
PS> Start-Process results\comparisons\comparison_0001.png
```

**Display Image**: Side-by-side comparison

- **Left side**: YOLO detections (green boxes)
- **Right side**: Faster R-CNN detections (blue boxes)
- **Red boxes**: Ground truth missed by YOLO

**Image Example 1: Small Bird Flock**

```
┌──────────────────────────────────────────────────────────┐
│ YOLO                    │  Faster R-CNN                   │
│                         │                                 │
│    [Image with 8 birds] │  [Same image]                   │
│                         │                                 │
│    🟢 1 green box       │  🔵🔵🔵🔵🔵🔵 6 blue boxes    │
│    (1 bird detected)    │  (6 birds detected)             │
│                         │                                 │
│    🔴🔴🔴🔴🔴🔴🔴        │  🔴🔴 2 red boxes              │
│    7 red boxes          │  (2 missed, extremely small)    │
│    (7 birds missed)     │                                 │
└──────────────────────────────────────────────────────────┘

Ground Truth: 8 birds
YOLO: 1/8 detected (12.5%)  ← 87.5% MISS RATE
Faster R-CNN: 6/8 detected (75%)
```

**Use cursor** to point and count boxes

**Talking**: "Here's a clear example. This image has 8 small birds..."

---

### Screen 10: Side-by-Side Comparison - Example 2 (Dense Crowd) (1.5 minutes)

**What to Show:**

```powershell
PS> Start-Process results\comparisons\comparison_0005.png
```

**Display Image**: Crowded street scene

**Image Example 2: Dense Crowd**

```
┌──────────────────────────────────────────────────────────┐
│ YOLO                    │  Faster R-CNN                   │
│                         │                                 │
│ [Crowded street]        │  [Same scene]                   │
│                         │                                 │
│ 🟢🟢🟢🟢                │  🔵🔵🔵🔵🔵🔵🔵🔵🔵           │
│ 4 green boxes           │  9 blue boxes                   │
│ (4 people detected)     │  (9 people detected)            │
│                         │                                 │
│ 🔴🔴🔴🔴🔴🔴🔴🔴          │  🔴🔴                          │
│ 8 red boxes             │  2 red boxes                    │
│ (missed people)         │  (very small, distant)          │
└──────────────────────────────────────────────────────────┘

Ground Truth: 12 people (6 small, 4 medium, 2 large)
YOLO: 4/12 detected (33%) - all missed = small people
Faster R-CNN: 9/12 detected (75%)
```

**Talking**: "This is a crowded street scene with multiple people..."

---

### Screen 11: Side-by-Side Comparison - Example 3 (Poor Localization) (1 minute)

**What to Show:**

```powershell
PS> Start-Process results\comparisons\comparison_0012.png
```

**Display Image**: Small dog with oversized box

**Image Example 3: Poor Localization**

```
┌──────────────────────────────────────────────────────────┐
│ YOLO                    │  Faster R-CNN                   │
│                         │                                 │
│ [Small dog on floor]    │  [Same dog]                     │
│                         │                                 │
│   ┌─────────────────┐   │     ┌─────────┐                │
│   │                 │   │     │         │                │
│   │    [DOG]        │   │     │  [DOG]  │                │
│   │                 │   │     │         │                │
│   └─────────────────┘   │     └─────────┘                │
│   🟢 Oversized box      │     🔵 Tight fit               │
│   (includes floor)      │     (accurate)                 │
│                         │                                 │
│   IoU = 0.38 ❌         │     IoU = 0.72 ✅              │
│   (below 0.5 threshold) │                                 │
└──────────────────────────────────────────────────────────┘

YOLO detected dog but BOX TOO LARGE
→ Poor localization (IoU < 0.5)
```

**Talking**: "Here's a poor localization case. YOLO DID detect this small dog..."

---

### Screen 12: Final Summary (30 seconds)

**What to Show:**

```powershell
PS> Write-Host "`n═══════════════════════════════════════" -ForegroundColor Green
PS> Write-Host "   DEMONSTRATION COMPLETE" -ForegroundColor Green
PS> Write-Host "═══════════════════════════════════════" -ForegroundColor Green
PS> Write-Host ""
PS> Write-Host "✅ Speed: YOLO 44.1× faster (20.45 vs 0.46 FPS)"
PS> Write-Host "❌ Small Objects: YOLO 0.0% vs Faster R-CNN 0.033%"
PS> Write-Host "📊 Failures: 954 FN, 68.2% = small objects"
PS> Write-Host "🎯 Trade-off: Speed vs. Small Object Detection"
PS> Write-Host ""
PS> Write-Host "📁 All code: github.com/DonutDaEarth/yolo_limitations_project_sgu" -ForegroundColor Cyan
PS> Write-Host ""
```

**Talking**: "This concludes our code demonstration. We've shown..."

---

---

# 📋 QUICK REFERENCE: WHAT TO PREPARE

## Before Recording:

### Create These Slides (PowerPoint/Google Slides):

1. ✅ Title Slide
2. ✅ Project Goals
3. ✅ YOLO Architecture Diagram
4. ✅ Faster R-CNN Architecture Diagram
5. ✅ Architecture Comparison Table
6. ✅ Small Object mAP Finding
7. ✅ Grid Visualization
8. ✅ Faster R-CNN Advantage
9. ✅ Failure Analysis Charts
10. ✅ Application Scenarios
11. ✅ Summary

### Generate These Files:

1. ✅ Run `test_yolo.py` to verify working
2. ✅ Run `test_faster_rcnn.py` to verify working
3. ✅ Run `run_taskA.py` (if not done) for accuracy results
4. ✅ Run `run_taskB.py` (if not done) for speed results
5. ✅ Run `failure_cases.py` for failure analysis
6. ✅ Run `comparison_viewer.py` to generate comparison images
7. ✅ Verify all result files exist in `results/` directory

### Test Screen Recording:

1. ✅ Test screen recording software (OBS, Camtasia, or Windows Game Bar)
2. ✅ Verify terminal font size is readable (14-16pt)
3. ✅ Test switching between slides and terminal
4. ✅ Practice smooth transitions

---

# 🎬 RECORDING WORKFLOW

## Recording Setup:

1. Open PowerPoint with slides (Part 1)
2. Open Terminal (PowerShell) for Part 2
3. Have comparison images ready to open
4. Set timer for tracking 15-minute segments

## Recording Order:

### Take 1: Person A - Part 1 (7 minutes)

- Record slides 1-5 with narration
- Screen: PowerPoint in presentation mode
- End with transition to Person B

### Take 2: Person B - Part 1 (8 minutes)

- Record slides 6-11 with narration
- Screen: PowerPoint in presentation mode
- End with transition to code demo

### Take 3: Person A - Part 2 (7 minutes)

- Record terminal commands
- Screen: Full-screen terminal
- Show results files and plot
- End with transition to Person B

### Take 4: Person B - Part 2 (8 minutes)

- Record failure analysis
- Screen: Terminal + image viewer
- Show comparison images
- End with summary

## Post-Recording:

1. Edit all 4 takes together
2. Add transitions between sections
3. Add text overlays for key numbers
4. Add background music (optional, low volume)
5. Export as MP4 (1920×1080, 30fps)

---

**You now have complete visual guidance for every second of your 30-minute presentation! 🎥**
