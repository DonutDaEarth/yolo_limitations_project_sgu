# Presentation Outline: Quantifying YOLO Limitations

**Duration:** 15 minutes (strict)  
**Team:** [Names]  
**Date:** [Date]

---

## Slide 1: Title Slide (30 seconds)

**Content:**

- Project Title: "Quantifying the Limitations of Single-Shot Detectors"
- Team Members: [Names]
- Course: Pattern Recognition & Image Processing
- Date

**Speaker Notes:**

- Brief greeting
- State project goal: Challenge YOLO's "high accuracy" claim

---

## Slide 2: Problem Statement (1 minute)

**Content:**

- Target Statement in large text:
  > "Single-shot detectors like YOLO prioritize efficiency and real-time performance while **maintaining high accuracy**, making them suitable for **many practical applications**."
- Question: "Is this always true?"
- Our approach: Quantitative + Qualitative analysis

**Visual:**

- Statement highlighted with question marks

---

## Slide 3: Models & Dataset (1 minute)

**Content:**
**Models Compared:**

- YOLOv8n (Single-stage)
- Faster R-CNN ResNet-50 (Two-stage)

**Dataset:**

- COCO 2017 Validation
- Focus: Small objects (< 32² pixels)
- 500 images evaluated

**Visual:**

- Side-by-side model architectures (simple diagrams)
- Sample COCO images with small objects

---

## Slide 4: Methodology Overview (1.5 minutes)

**Content:**
**Task A: Accuracy Challenge**

- Metrics: mAP@[0.5:0.95], mAP(Small), mAP(Medium/Large)
- Identify failure modes

**Task B: Speed Benchmark**

- Measure FPS on 500 images
- Speed-accuracy trade-off

**Hardware:**

- [Your GPU]
- CUDA [version]

**Visual:**

- Simple flowchart of methodology

---

## Slide 5: Results - Detection Performance (1.5 minutes)

**Content:**
**Table: Overall Metrics**

| Metric         | YOLO     | Faster R-CNN | Difference |
| -------------- | -------- | ------------ | ---------- |
| mAP@[0.5:0.95] | X.X%     | Y.Y%         | **+Z.Z%**  |
| **mAP(Small)** | **X.X%** | **Y.Y%**     | **+Z.Z%**  |
| mAP(Medium)    | X.X%     | Y.Y%         | +Z.Z%      |
| mAP(Large)     | X.X%     | Y.Y%         | +Z.Z%      |

**Key Finding:** Faster R-CNN achieves **Z%** higher mAP on small objects

**Visual:**

- Highlight mAP(Small) row in red
- Bar chart comparing mAP values

---

## Slide 6: Results - Speed vs Accuracy (1.5 minutes)

**Content:**
**Speed Comparison:**

- YOLO: [X] FPS ([Y] ms per image)
- Faster R-CNN: [Z] FPS ([W] ms per image)
- Speed-up: **Nx faster**

**Trade-off:**

- For every [A]% accuracy loss, YOLO gains [B]x speed

**Visual:**

- Speed-Accuracy Trade-off Plot (your generated plot)
- YOLO in red zone (high speed), Faster R-CNN in green zone (high accuracy)

---

## **Slides 7-12: FAILURE MODES (8 MINUTES - CRITICAL!)** ⚠️

## Slide 7: Failure Case Statistics (1 minute)

**Content:**
**Analysis of 500 Images:**

- Total YOLO failures identified: [N]
- False Negatives: [X]
- Poor Localization: [Y]

**By Object Size:**

- Small objects: **[Z]%** of failures
- Medium objects: [A]%
- Large objects: [B]%

**Key Insight:** Failures concentrated on small objects!

**Visual:**

- Pie chart of failure distribution by size

---

## Slide 8-9: Failure Examples 1-3 (2 minutes)

**Content:**
**Each slide shows 1-2 failure cases with 3-panel view:**

- Left: Ground Truth
- Center: YOLO (failed)
- Right: Faster R-CNN (success)

**Case 1: Dense Small Object Cluster**

- Scenario: Crowded scene with 15+ small objects
- YOLO: Detected 6/15 (40%)
- Faster R-CNN: Detected 13/15 (87%)

**Case 2: Small Person Detection**

- Object size: 24×18 pixels
- YOLO: Missed (IoU = 0.1)
- Faster R-CNN: Detected (IoU = 0.7)

**Case 3: Overlapping Small Objects**

- Multiple objects in single grid cell
- YOLO: NMS suppressed detections
- Faster R-CNN: Separate proposals for each

**Visual:**

- Large, clear images from failure*case*\*.png files
- Annotate why YOLO failed

---

## Slide 10-11: More Failure Examples 4-7 (2 minutes)

**Content:**
Similar format, showcasing:

- Case 4: Small objects at image edges
- Case 5: Low contrast small objects
- Case 6: Dense cluster of medium objects
- Case 7: Partial occlusion

**Visual:**

- 2 cases per slide
- Brief captions explaining each

---

## Slide 12: WHY YOLO Fails - Architecture Analysis (3 minutes) ⭐

**Content:**
**Root Cause #1: Grid-Based Detection**

- YOLO divides image into fixed grid (e.g., 20×20)
- Each cell predicts limited boxes
- Problem: Multiple small objects → one grid cell → suppression

**Root Cause #2: Feature Map Resolution**

- Deep backbone reduces resolution significantly
- Small objects become 1-2 pixels on feature map
- Insufficient spatial information

**Root Cause #3: NMS Behavior**

- Aggressive NMS (IoU 0.45) for speed
- Over-suppresses overlapping small detections
- Faster R-CNN: Two-stage refinement + conservative NMS

**Visual:**

- Architecture diagrams showing:
  - YOLO grid on sample image
  - Feature map resolution comparison
  - NMS effect illustration

**Speaker Notes (Critical):**
"This is the key insight - YOLO's speed comes from these architectural choices, but they directly cause small object failures. The grid-based approach fundamentally limits its ability to handle dense small objects."

---

## Slide 13: Practical Implications (1.5 minutes)

**Content:**
**When YOLO Works Well:**
✅ Autonomous driving (speed critical, objects typically large)
✅ Real-time surveillance (acceptable miss rate)
✅ Robotics (low-latency requirements)

**When YOLO Fails:**
❌ Medical imaging (small tumors, accuracy critical)
❌ Quality inspection (small defects must be detected)
❌ Satellite imagery (predominantly small objects)

**Trade-off Decision Framework:**

- Speed requirement > Accuracy: Choose YOLO
- Accuracy requirement > Speed: Choose Faster R-CNN

**Visual:**

- Icons for each application
- Green/red indicators

---

## Slide 14: Conclusion (1 minute)

**Content:**
**Key Findings:**

1. YOLO's "high accuracy" is **contextual** - breaks down on small objects
2. Speed-accuracy trade-off: [N]x faster at cost of [M]% mAP
3. [X]% of failures on small objects demonstrate architectural limitations

**Final Answer:**
The statement is **partially incorrect** - YOLO maintains high accuracy only on specific object sizes and scenarios, NOT "many practical applications" universally.

**Future Work:**

- Test larger YOLO variants (YOLOv8m, YOLOv8x)
- Evaluate on other challenging datasets
- Explore hybrid approaches

---

## Slide 15: Q&A (Remaining time)

**Content:**

- "Questions?"
- Team contact information

**Anticipated Questions & Prep Answers:**

1. **"Why not test YOLOv8x (larger model)?"**

   - Answer: Wanted to show worst-case scenario with smallest variant. Larger variants likely improve but don't fundamentally solve grid-based limitations.

2. **"Can YOLO be fine-tuned for small objects?"**

   - Answer: Yes, but this requires training data and computational resources. Our goal was to evaluate pre-trained models as claimed "suitable for many applications."

3. **"What about real-time requirements?"**

   - Answer: Excellent point - for applications requiring >30 FPS, YOLO is essential despite accuracy trade-off. This validates our conclusion: suitability is context-dependent.

4. **"How did you select the 10 failure cases?"**
   - Answer: Automated analysis of all predictions, selected based on severity (IoU difference), size diversity, and failure type variety.

---

## Timing Breakdown (Strict 15 Minutes)

| Slides              | Duration  | Cumulative |
| ------------------- | --------- | ---------- |
| 1-3                 | 2.5 min   | 2.5 min    |
| 4-6                 | 4.5 min   | 7 min      |
| **7-12 (Failures)** | **8 min** | **15 min** |
| 13-14               | 2.5 min   | 17.5 min   |

**Strategy:** If running over time, cut:

- Slide 4 (methodology) to 1 min
- Reduce failure examples to 6 instead of 7
- Never cut Slide 12 (WHY analysis) - this is critical!

---

## Presentation Tips

### Visual Design

- **Font:** Large (24pt+ for body, 36pt+ for titles)
- **Colors:** YOLO in red, Faster R-CNN in blue, highlights in yellow
- **Images:** High resolution failure case images, 2-3 per slide max
- **Charts:** Clear labels, legends, no clutter

### Delivery

- **Practice:** Rehearse 3+ times, time yourself
- **Transitions:** Smooth handoffs between speakers
- **Emphasis:** Spend 8+ minutes on failures (Slides 7-12)
- **Engagement:** Point to specific examples in images
- **Confidence:** You have data - let results speak

### Technical Setup

- **Backup:** PDF + PowerPoint versions
- **Demo Ready:** Have comparison viewer pre-loaded
- **Laser Pointer:** For highlighting failure cases in images
- **Notes:** Print speaker notes as backup

---

## Speaker Assignments (Example for 3-person team)

**Speaker 1:** Slides 1-6 (Introduction, Methods, Results)
**Speaker 2:** Slides 7-12 (Failure Modes - MOST IMPORTANT)
**Speaker 3:** Slides 13-15 (Implications, Conclusion, Q&A)

Adjust based on team size and strengths.

---

**Remember:** The goal is to CHALLENGE the statement, not praise YOLO. Focus on WHERE and WHY it breaks down!
