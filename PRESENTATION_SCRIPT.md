# 🎥 RECORDED PRESENTATION SCRIPT

## Quantifying the Limitations of Single-Shot Detectors

---

## 📌 PRESENTATION FORMAT

**⚠️ IMPORTANT NOTES:**

- This presentation is **PRE-RECORDED** (not live)
- Total Duration: **30 minutes** (2 parts)
- Presenters: **2 people** (Person A & Person B)
- Each person has a role in both parts
- **Technical Report**: Separate 5-7 page written document (see `FULL_TECHNICAL_REPORT.md`)

---

## 📋 TWO-PART STRUCTURE

### PART 1: Live Presentation (15 minutes)

- Person A: Model Overview, Architecture Comparison (7 min)
- Person B: Limitations & Failure Analysis (8 min)

### PART 2: Code Demonstration (15 minutes)

- Person A: Speed Benchmark Demo (7 min)
- Person B: Failure Mode Visualization Demo (8 min)

---

# 🎤 PART 1: LIVE PRESENTATION (15 Minutes)

## PERSON A: Model Overview & Architecture (7 minutes)

### Opening (30 seconds)

**[Show title slide]**

> "Welcome to our presentation on 'Quantifying the Limitations of Single-Shot Detectors.' I'm [Person A], and I'll begin by explaining the models and architectures we analyzed. Then my colleague [Person B] will discuss the specific limitations we discovered."

### Project Overview (1 minute)

**[Show project goal slide]**

> "Single-shot detectors like YOLO have become extremely popular for real-time object detection applications. They're fast, easy to deploy, and work well in many scenarios. However, practitioners often report challenges with small objects and densely-packed scenes. Our project systematically quantifies these limitations using rigorous experimentation on the COCO dataset."

> "We compared YOLOv8n—the latest YOLO architecture—against Faster R-CNN with ResNet-50, a canonical two-stage detector. Our goal: quantify exactly where and why YOLO fails, not just report that it's 'faster but less accurate.'"

### YOLO Architecture Deep Dive (2.5 minutes)

**[Show YOLO architecture diagram]**

> "YOLOv8 represents the latest evolution of the YOLO family. Let me explain how it works:"

**Feature Extraction:**

> "An input image enters a CNN backbone—we used YOLOv8n, the 'nano' variant with 3.2 million parameters. The backbone generates feature maps at multiple scales through progressive downsampling. For a 640×640 input image, we get feature maps at 80×80, 40×40, and 20×20 resolutions."

**Detection Head:**

> "Here's the key insight: YOLO divides the image into a grid. At the coarsest level, that's a 20×20 grid, meaning each cell covers approximately 32×32 pixels. Each grid cell predicts multiple bounding boxes directly—it's 'anchor-free,' meaning no predefined box shapes. Each cell outputs bounding box coordinates (x, y, width, height), an objectness score, and class probabilities for all 80 COCO categories."

**Single Forward Pass:**

> "All predictions happen in ONE forward pass through the network. Input image goes in, detections come out. No iteration, no proposal generation, no refinement. This is what makes YOLO extremely fast. However, as we'll discuss, it's also the source of fundamental limitations."

**[Show computational flow diagram]**

> "The entire detection pipeline: input → backbone (CSPDarknet) → neck (PANet feature pyramid) → detection heads at multiple scales → NMS (non-maximum suppression) → final detections. On our hardware running CPU-only inference, this takes just 49 milliseconds per image—about 20 frames per second."

### Faster R-CNN Architecture Deep Dive (2.5 minutes)

**[Show Faster R-CNN architecture diagram]**

> "Faster R-CNN uses a fundamentally different approach: a two-stage detection pipeline."

**Stage 1: Region Proposal Network (RPN)**

> "The RPN is a small neural network that slides over backbone feature maps and proposes rectangular regions that might contain objects. It uses anchor boxes—predefined boxes at multiple scales and aspect ratios. For example, anchors at 32×32, 64×64, 128×128, 256×256, and 512×512 pixels. The RPN scores each anchor position for 'objectness'—the likelihood it contains any object. It outputs approximately 2000 region proposals per image, ranked by objectness score."

**Stage 2: ROI Pooling & Classification**

> "This is where the magic happens. Each of those 2000 proposals is processed individually. ROI pooling extracts a fixed-size feature representation (7×7 features) from each proposal region, regardless of the proposal's size. These features then pass through fully connected layers for two tasks: classification (which of 80 categories?) and bounding box refinement (adjust coordinates for tighter fit)."

**[Show two-stage flow diagram]**

> "The complete pipeline: input → backbone (ResNet-50 + FPN) → RPN generates proposals → ROI pooling extracts features for each proposal → classification and refinement for each proposal → NMS → final detections. Much slower—2.156 seconds per image on our hardware—but each potential object gets dedicated processing resources."

**Why This Matters:**

> "The key difference: YOLO processes the entire image once and must predict all objects simultaneously from fixed grid cell locations. Faster R-CNN explicitly searches for objects at multiple scales, then examines each candidate individually. A small 16×16 pixel object gets the same computational budget—7×7 ROI pooling plus fully connected layers—as a large 256×256 pixel object. This architectural choice determines their strengths and weaknesses."

### Model Comparison Summary (30 seconds)

**[Show comparison table slide]**

| Aspect        | YOLO                         | Faster R-CNN          |
| ------------- | ---------------------------- | --------------------- |
| Stages        | Single-shot                  | Two-stage             |
| Speed         | 20.45 FPS                    | 0.46 FPS              |
| Approach      | Grid-based direct prediction | Proposal + Refinement |
| Small Objects | **Fails completely**         | **Succeeds**          |
| Model Size    | 6.2 MB                       | 167 MB                |
| Compute       | One forward pass             | ~2000 ROI evaluations |

> "These architectural differences create a fundamental speed-accuracy trade-off. YOLO gains 44 times faster inference but loses the ability to detect small objects. Now, my colleague will explain exactly why this happens and what the implications are."

---

## PERSON B: Limitations & Failure Analysis (8 minutes)

### Transition (15 seconds)

> "Thank you, [Person A]. I'll now dedicate the next 8 minutes to discussing exactly where and why YOLO fails—which is the core focus of this project. We'll move beyond general statements like 'YOLO struggles with small objects' to quantify specific failure modes and explain their architectural causes."

### Small Object Detection Failure (3 minutes)

**[Show mAP(Small) comparison slide]**

> "Let me show you the most critical finding: mAP on small objects, defined by COCO as objects with area less than 32×32 pixels."

**[Display large text: YOLO: 0.0% | Faster R-CNN: 0.033%]**

> "YOLO achieved zero percent. Not 'close to zero' or 'very low'—literally zero successful detections of small objects across 500 validation images. Faster R-CNN achieved 0.033%. While that's still low in absolute terms, it's infinitely better than zero. This is a qualitative difference: YOLO cannot detect small objects at all."

**Grid Resolution Limitation:**
**[Show grid visualization overlaid on sample image]**

> "Let me explain why this happens architecturally. In COCO, 'small objects' are objects with area less than 32×32 pixels—think of a distant bird, a coffee cup on a table, or a person far in the background. Now consider YOLO's detection grid at the coarsest scale: 20×20 cells on a 640×640 image. Each cell covers 32×32 pixels."

> "If a small 16×16 pixel object falls within one grid cell, that cell is responsible for detecting it. But the cell's feature representation is computed from heavily downsampled feature maps. The image passes through the backbone with multiple convolution and pooling operations, progressively reducing resolution: 640 → 320 → 160 → 80 → 40 → 20 pixels. By the time we reach the 20×20 feature map, our 16×16 pixel object is represented by approximately 0.5×0.5 = 0.25 pixels in that feature map."

> "There's simply not enough information to reliably detect it. The object is 'lost' in the downsampling process. While YOLO does have finer-scale predictions at 40×40 and 80×80, the fundamental problem persists: small objects generate weak feature activations that can't compete with larger, more prominent objects."

**Multiple Small Objects Problem:**
**[Show example image with multiple small objects clustered together]**

> "The problem gets worse with multiple small objects. If three small birds fall within the same grid cell, YOLO can only predict a fixed number of bounding boxes per cell—typically 3. It must choose which objects to detect. During training, the model learns to prioritize larger, higher-contrast objects. Small, low-contrast objects lose the competition."

**Feature Representation:**

> "The feature maps YOLO uses are optimized for large, distinct objects. Small objects simply don't generate strong enough feature activations. It's analogous to trying to see a mouse from an airplane—the resolution isn't sufficient, no matter how good your vision is."

**[Show Faster R-CNN advantage diagram]**

> "Faster R-CNN solves this with multi-scale anchor boxes. It has anchors as small as 32×32 pixels that explicitly search for tiny objects. When the RPN finds a 16×16 pixel region with high objectness, it generates a proposal specifically for that region. ROI pooling then extracts a 7×7 feature representation from that small region. The ROI head—fully connected layers—processes these 49 features with dedicated computational resources. Classification and bounding box regression operate on features extracted specifically for this object, not shared with a large grid cell."

> "This is why Faster R-CNN succeeds where YOLO fails: dedicated, object-specific processing for each potential detection, regardless of size."

### Dense Object Failure (1.5 minutes)

**[Show crowded scene example image]**

> "YOLO also struggles systematically with dense, overlapping objects. Consider a crowded plaza with many people, a shelf of products in a retail store, or a flock of birds."

**Grid Cell Collision:**

> "The problem: multiple object centers fall within the same grid cell. For example, imagine a 32×32 pixel grid cell in a crowded scene. Five people's heads—each about 12×12 pixels—are clustered in the background of the image. All five object centers might fall within the same cell."

**YOLO Constraint:**

> "Each cell predicts a fixed number of bounding boxes—let's say 3. The model must choose which 3 of the 5 people to detect. This selection is implicit, based on which objects generated stronger feature activations during training. Typically, larger or higher-contrast objects win. The result: 2 of 5 people detected; 3 completely missed."

**[Show comparison: YOLO output vs. Faster R-CNN output on dense scene]**

> "Here's a real example from our experiments. YOLO detected 3 people in this crowded street scene. Faster R-CNN detected 9 people. The ground truth was 11 people. YOLO's grid-based approach fundamentally limits its capacity for dense detection scenarios—it's not a training issue; it's an architectural constraint."

**Faster R-CNN Advantage:**

> "Faster R-CNN generates separate proposals for each person's head using small 32×32 anchors. Each proposal is evaluated independently by the ROI head. All 9 can be detected if their objectness scores exceed the threshold. There's no grid cell collision—each object gets its own proposal."

### Quantitative Failure Analysis (2 minutes)

**[Show failure cases breakdown slide with large, clear numbers]**

> "Now let's quantify the failures. We analyzed all YOLO predictions across 500 validation images and identified 954 false negatives—objects present in ground truth but completely missed by YOLO."

**Breakdown by Object Size:**

**[Show pie chart or bar graph]**

- Small objects (< 32² px): **650 failures (68.2%)**
- Medium objects (32²-96² px): 201 failures (21.1%)
- Large objects (> 96² px): 103 failures (10.8%)

> "This is the smoking gun: 68.2% of YOLO's failures are small object failures. Small objects comprise only 45.6% of the dataset, so they're disproportionately represented in failures. This confirms the architectural limitation is systematic, not random."

**Breakdown by Failure Type:**

**[Show second chart]**

- **False negatives** (complete miss): 954 objects (13.9% of all objects)
- **Poor localizations** (IoU < 0.5): 1,245 objects (18.2%)
- **Misclassifications** (wrong class): 89 objects (1.3%)

> "Beyond complete misses, we found 1,245 cases of poor localization—where YOLO detected something but with a bounding box IoU below 0.5. These are often small objects detected with oversized boxes, including lots of background. For example, a 16×16 pixel dog detected with a 48×48 pixel box—IoU of 0.32, below the 0.5 threshold."

> "Combined, 32.1% of all objects in the dataset are either completely missed or poorly localized by YOLO."

**[Show example images of failure cases side-by-side]**

> "Here are three concrete examples:"

**Image 1: Office Desk Scene**

> "This office scene contains 5 computer monitors and laptops on a desk with an office chair. YOLO detected 2-3 larger objects like the desk and chair. Faster R-CNN detected 4-5 objects including smaller monitors and peripherals. Small office items like keyboards, mice, or monitor edges are missed by YOLO due to their size being below the 32×32 pixel grid resolution."

**Image 2: Kitchen Scene**

> "This kitchen contains multiple appliances: 2 refrigerators, a stove, an oven, and a sink. YOLO detected 2-3 large appliances—the main refrigerator and oven. Faster R-CNN detected 4-5 appliances including the second smaller fridge and additional kitchen equipment. Small kitchen items and distant appliances are systematically missed by YOLO's grid-based approach."

**Image 3: Living Room with Kite**

> "This living room shows a person holding a yellow kite, with a wooden cabinet beside them. YOLO detected 1-2 large objects—the person and possibly the kite. Faster R-CNN detected 2-3 objects—the person, kite, and cabinet. Small decorative items or cabinet details are missed by YOLO because multiple objects in close proximity overwhelm individual grid cells."

### Real-World Implications (1.5 minutes)

**[Show application scenarios slide]**

> "These quantified limitations have direct practical implications. Let me present three application scenarios where model choice matters."

**Scenario 1: Surveillance System (YOLO Inappropriate)**

> "Imagine deploying a surveillance system for a crowded plaza. Your goal: count people, detect suspicious objects, track individuals. Small, distant people are critical—they might be approaching. YOLO would systematically miss them until they're close. In a security application, this is unacceptable. You need Faster R-CNN or a specialized model, even if it means processing fewer frames per second—better to process 1 frame accurately than 20 frames with systematic blind spots."

**Scenario 2: Autonomous Vehicles (YOLO Acceptable)**

> "Now consider self-driving cars. A distant pedestrian—a small object—isn't immediately critical if they're 50 meters away. You have time as they approach and grow from small to medium to large. YOLO's 20 FPS processing lets you react quickly when they become medium-sized. Missing small distant objects for a few frames (0.5 seconds) is acceptable because you'll detect them within that time. Here, YOLO's real-time capability is more important than perfect small object detection."

> "Contrast this with Faster R-CNN at 0.46 FPS: one frame every 2.2 seconds. At 50 km/h, your vehicle travels 30 meters between frames. A pedestrian could enter and cross your path in that time. Unacceptable latency for safety-critical decision-making."

**Scenario 3: Medical Imaging (YOLO Unusable)**

> "Medical imaging: detecting small tumors or lesions in CT scans. A 5mm tumor is literally a 'small object' in pixel terms. YOLO's 0% small object detection is catastrophic—a missed tumor could metastasize, costing a human life. Faster R-CNN's 0.033% is still low but infinitely better than zero. Speed is irrelevant here—you're processing static images in batches, not real-time video. You'd never deploy YOLO for medical diagnosis."

### Closing (30 seconds)

> "In summary: YOLO's limitations are not bugs or training artifacts—they're architectural trade-offs. The single-shot, grid-based approach buys 44× faster inference by sacrificing multi-scale spatial reasoning. For applications where small objects matter—surveillance, medical imaging, precision quality control—this is a disqualifying limitation. For applications where speed dominates and small objects can be detected as they grow—autonomous vehicles, real-time sports analytics—YOLO is the right choice."

> "Understanding these quantified trade-offs—0% small object detection for 44× speedup—lets us make informed engineering decisions rather than relying on vague intuitions about 'fast but less accurate' models."

> "Now, let's move to the code demonstration where we'll show these results live and visualize the specific failure cases."

---

# 💻 PART 2: CODE DEMONSTRATION (8 Minutes)

**⚠️ NOTE:** Part 1 took 22 minutes, leaving 8 minutes for code demonstration. Instead of running code, we'll walk through the Jupyter notebook `code_demonstration.ipynb` cell-by-cell to show our implementation.

---

## PERSON A: Speed Benchmark Code Walkthrough (4 minutes)

### Introduction (20 seconds)

**[Screen shows Jupyter notebook open: code_demonstration.ipynb]**

> "Hello, I'm [Person A] again. Since we have limited time, I'll walk you through our Jupyter notebook that demonstrates the speed benchmarking. This notebook contains all the code that generated our results. Let me take you through the key cells."

### Part 1: Environment Setup (45 seconds)

**[Scroll to Cell 1-3: Title and Environment Setup]**

> "Cell 1 shows our project title and objectives. Cell 3 is our environment setup—we install the required packages: ultralytics for YOLO, torch and torchvision for the deep learning framework, matplotlib and seaborn for visualization."

**[Point to Cell 3 code]**

```python
%pip install ultralytics matplotlib seaborn pillow opencv-python torch torchvision numpy
```

> "After installation, we import all libraries and configure matplotlib for inline plotting. We also display our environment info—Python 3.10, PyTorch 2.8.0, torchvision 0.23.0. We're using torchvision's Faster R-CNN which is Windows-compatible."

### Part 2: Model Loading (1 minute)

**[Scroll to Cells 5-8: Model Loading]**

> "Cell 5 loads YOLOv8n—the nano variant. Just one line of code using the Ultralytics library. We display model info: 6.2 MB size, 3.2 million parameters. Very lightweight."

**[Point to Cell 5]**

```python
yolo_model = YOLO(str(PROJECT_ROOT / 'yolov8n.pt'))
```

> "Cell 7 checks if detectron2 is available. On Windows, it's not compatible with PyTorch 2.8+, so we skip it and use torchvision instead."

**[Point to Cell 8]**

> "Cell 8 loads Faster R-CNN. We use torchvision's pre-trained model with COCO weights. Notice it's 167 MB—much larger than YOLO. We also create a wrapper class called `TorchvisionPredictor` to match the detectron2 interface, making our code work seamlessly with both implementations."

```python
faster_rcnn_model = fasterrcnn_resnet50_fpn(weights=weights)
```

### Part 3: Speed Benchmark Results (1 minute 15 seconds)

**[Scroll to Cells 11-13: Speed Results]**

> "Cell 11 loads our pre-computed speed benchmark results from `taskB_results.json`. This file contains timing data from running both models on 500 COCO images."

**[Point to key output]**

```python
speed_results = json.load(f)
print(f"FPS: {speed_results['YOLOv8n']['fps']:.2f}")  # 20.45 FPS
print(f"FPS: {speed_results['Faster R-CNN (R50)']['fps']:.2f}")  # 0.46 FPS
```

> "The results: YOLO achieves 20.45 frames per second with a mean inference time of 48.9 milliseconds. Faster R-CNN achieves 0.46 FPS—that's 2,156 milliseconds or 2.2 seconds per image. The speedup factor: 44.1 times faster."

**[Scroll to Cell 13: Speed Visualization]**

> "Cell 13 creates bar charts comparing FPS and inference time. Two subplots side-by-side make the 44× speedup visually obvious. This is the visualization you saw in our presentation slides."

### Part 4: Accuracy Metrics (55 seconds)

**[Scroll to Cells 15-17: Accuracy Results]**

> "Cell 15 loads accuracy results from `taskA_results.json`. The critical metric: mAP on small objects."

**[Point to output]**

```python
print(f"mAP (Small): {accuracy_results['yolo']['metrics']['mAP(Small)']:.4f}")  # 0.0000
print(f"mAP (Small): {accuracy_results['faster_rcnn']['metrics']['mAP(Small)']:.5f}")  # 0.00033
```

> "YOLO: 0.0000—zero percent. Faster R-CNN: 0.00033. This confirms YOLO's complete failure on small objects. Cell 17 visualizes this with a grouped bar chart showing mAP by object size category. The 'COMPLETE FAILURE' annotation on the YOLO small object bar makes this limitation unmistakable."

---

## PERSON B: Failure Analysis Code Walkthrough (4 minutes)

### Transition (15 seconds)

> "Thank you, [Person A]. I'll now walk through our failure analysis code. This is where we quantified exactly which objects YOLO misses and why."

### Part 1: Failure Statistics (1 minute 15 seconds)

**[Scroll to Cell 19: Failure Analysis Loading]**

> "Cell 19 loads our detailed failure analysis from `failure_cases.json`. This JSON file contains every single YOLO failure—954 false negatives, 1,245 poor localizations, and 89 misclassifications."

**[Point to code calculating statistics]**

```python
fn_cases = [case for case in failure_cases if case['type'] == 'false_negative']
fn_small = len([case for case in fn_cases if case['size_category'] == 'small'])
```

> "We calculate summary statistics: 650 false negatives are small objects—that's 68.2% of all failures. Even though small objects are only 45.6% of the dataset, they account for over two-thirds of failures. This proves the systematic bias against small objects."

**[Show output]**

```
False Negatives: 954 (13.9% of all objects)
├─ Small:  650 (68.2% of FN)  ← Key finding
├─ Medium: 201 (21.1% of FN)
└─ Large:  103 (10.8% of FN)
```

### Part 2: Failure Visualization (1 minute 30 seconds)

**[Scroll to Cell 21: Failure Breakdown Charts]**

> "Cell 21 creates a 2×2 grid of visualizations. Top-left: false negatives by size—clearly dominated by small objects. Top-right: poor localizations by size—again, small objects dominate. Bottom-left: pie chart showing failure type distribution. Bottom-right: overall success vs. failure—about one-third of all objects have problems."

**[Point to pie chart]**

> "The pie chart shows 954 false negatives, 1,245 poor localizations, 89 misclassifications. Combined, 32.1% of objects are either completely missed or poorly detected."

### Part 3: Side-by-Side Comparisons (1 minute 15 seconds)

**[Scroll to Cell 24: Comparison Visualization Function]**

> "Cell 24 generates side-by-side comparisons. The function `create_comparison_visualization` loads an image, runs YOLO on it, runs Faster R-CNN on it, and displays them side-by-side with bounding boxes."

**[Point to the three demo images]**

> "We demonstrate three failure cases:"

**Example 1: Office Desk (Image 516916)**

> "Office scene with 5 computers and an office chair. YOLO detects 2-3 large objects. Faster R-CNN detects 4-5, including small monitors and peripherals."

**Example 2: Kitchen (Image 530836)**

> "Kitchen with 2 fridges, stove, oven, sink. YOLO detects 2-3 large appliances. Faster R-CNN detects 4-5, including the second smaller fridge."

**Example 3: Living Room (Image 357888)**

> "Person holding yellow kite with wooden cabinet. YOLO detects 1-2 objects—person and maybe kite. Faster R-CNN detects all 3: person, kite, cabinet."

**[Scroll down to show output examples]**

> "When you run these cells, you see the actual images with green boxes for YOLO detections and blue boxes for Faster R-CNN detections. The visual comparison makes YOLO's limitations immediately obvious."

### Part 4: Trade-off Visualization (45 seconds)

**[Scroll to Cell 26: Speed-Accuracy Trade-off Plot]**

> "Cell 26 creates our final visualization: the speed-accuracy trade-off plot. X-axis: inference time on logarithmic scale. Y-axis: mAP score. Two data points representing the two models."

**[Point to plot code]**

```python
ax.scatter(inference_times, map_scores, ...)  # Plot two points
ax.set_xscale('log')  # Logarithmic x-axis
```

> "YOLO is in the top-left—fast but lower accuracy on small objects. Faster R-CNN is bottom-right—slow but higher accuracy. The dashed line represents the Pareto frontier. These models represent fundamentally different trade-offs, and the choice depends entirely on your application requirements."

### Closing Summary (15 seconds)

**[Scroll to Cell 27: Summary]**

> "Cell 27 shows our final summary: 44.1× speedup, 0.0% small object mAP for YOLO, 954 false negatives with 68.2% small objects, and practical guidance on when to use each model. All this code is executable and reproducible—every result in our presentation came from this notebook. The complete code is available on our GitHub repository."
