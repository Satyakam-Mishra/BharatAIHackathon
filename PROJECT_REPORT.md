# Final Project Report: Real-Time On-Device AI Visual Inspection and Advisory System for Manufactured PCBs

**Hackathon Track:** Bharat AI Hackathon - Edge AI Track  
**Project Title:** On-Device PCB Quality Control and Operator Advisory System  
**Team Members:** Satyakam Mishra, Aryan Lahoti, Kavya Soni  
**Project Guide:** Adithya Kumar  
**Target Hardware:** Raspberry Pi 5 (Broadcom BCM2712 Quad-Core Arm Cortex-A76 @ 2.4 GHz, 8GB RAM)  

---

## 1. Executive Summary and Problem Statement

Printed circuit board (PCB) manufacturing requires continuous optical inspection to detect surface anomalies prior to component assembly and wave soldering. Conventional Automated Optical Inspection (AOI) setups depend either on brittle rule-based template comparison or on cloud-connected deep learning APIs. Cloud-based inspection introduces latency bottlenecks, recurring network costs, data security concerns, and downtime when factory connectivity is disrupted. Furthermore, raw detection bounding boxes do not provide floor operators with actionable rework instructions.

This project implements an autonomous, on-device edge AI system that runs locally without cloud dependence. The system combines:
1. An input pre-inspection binary gate (`pcb_gate`) that prevents false inference on non-PCB objects.
2. A fine-tuned YOLOv8n detector operating over high-resolution sliding-window tiles to resolve microscopic defects without downsampling loss.
3. A deterministic spatial and physical severity grading engine.
4. An on-device Llama 3.2 1B Instruct language model deployed via Meta ExecuTorch with XNNPACK Arm acceleration, producing structured rework directives.
5. An automated post-generation safety validator ensuring zero hallucination.
6. A local SQLite industrial ledger storing inspection records and hardware telemetry (CPU, RAM, thermals).

```
[ Camera / Input Frame ]
           │
           ▼
[ Pre-Inspection Gate (YOLOv8n-cls 224x224) ]
     ├── p(PCB) < 0.60 ──► Reject / Await PCB
     └── p(PCB) >= 0.60
           │
           ▼
[ Tiled Defect Detector (YOLOv8n 640x640) ]
[ Overlapping Stride: 512px | Coordinate Mapping ]
           │
           ▼
[ Duplicate Box Merging (NMS > 50% Overlap) ]
           │
           ▼
[ Deterministic Grading Engine ] ──► Instant PASS / FAIL Verdict
  - Spatial Localization (3x3 Grid)
  - Base Severity + Size Escalation
           │
           ▼
[ On-Device Llama 3.2 1B Advisory (ExecuTorch) ]
[ Grounded SOP Prompting + Regex Validation Guardrail ]
           │
           ▼
[ Streamlit Operator Console & SQLite Telemetry Log ]
```

---

## 2. Methodology

### 2.1 Dataset and Defect Taxonomy
The defect detection model is trained on the Peking University (PKU) Market PCB Defect Dataset, augmented via Roboflow into 3,326 labeled images across 6 primary industrial defect classes:
- `missing_hole`: Un-drilled vias or pin pads.
- `mouse_bite`: Edge nibbles or track necking causing impedance degradation.
- `open_circuit`: Complete discontinuity in conductive copper traces.
- `short`: Unintended copper bridging between independent nets.
- `spur`: Conductor spikes protruding toward adjacent tracks.
- `spurious_copper`: Extraneous isolated copper patches from imperfect etching.

Augmentations include random horizontal/vertical flips, 90-degree rotations, +/-5 degree micro-rotations, +/-20% brightness variance, 0.22% salt-and-pepper noise simulation, and Gaussian bounding box jitter (0-2 px) to model industrial conveyor vibrations and variable factory lighting.

### 2.2 Pre-Inspection Gating (`pcb_gate`)
To conserve compute cycles on edge processors, a lightweight YOLOv8n classifier (224x224, 2.8 MB) acts as an input gate before running full multi-tile defect detection:
- Evaluates input frames against a probability threshold of p(pcb) >= 0.60.
- Benchmark evaluation across 359 test images (195 non-PCB everyday images + 164 PCB images) achieved 100% accuracy (359/359).
- Bypasses empty conveyor passes, reducing unnecessary CPU utilization.

### 2.3 High-Resolution Tiled Inference Engine
Downsampling full-resolution PCB images (e.g. 3034x1586 px) directly to 640x640 causes thin traces and small micro-defects (such as 10-pixel spurs) to disappear. The vision engine resolves this via overlapping sliding windows:
1. **Tiling:** Divides the frame into 640x640 px tiles using a 512 px stride, creating a 128 px overlap between adjacent tiles.
2. **Coordinate Projection:** Detections from local tile coordinates are mapped back to absolute image coordinates: $[(x + x_0)/z, (y + y_0)/z]$, accounting for camera zoom $z$.
3. **Cross-Tile Deduplication:** Custom Non-Maximum Suppression merges duplicate boxes spanning tile seams when overlap exceeds 50% of the smaller box, retaining the highest confidence score.
4. **Adaptive Digital Zoom:** Supports 1.0x to 3.0x magnification for situations where cameras are mounted at greater working distances.

### 2.4 Deterministic Grading and Safety Guardrails
In quality control, critical decisions must be deterministic:
- **Spatial Grid:** A 3x3 coordinate partition maps defects into human-readable sectors (`top-left`, `top-centre`, `centre`, `bottom-right`, etc.).
- **Severity Classification:**
  - High: `open_circuit`, `short`, `missing_hole` (direct electrical failure).
  - Medium: `mouse_bite`, `spur` (clearance or impedance risks).
  - Low: `spurious_copper` (cosmetic or isolated non-functional copper).
- **Dynamic Size Scaling:** Any defect whose longest dimension exceeds 3% of the image's longest dimension is escalated by one grade (e.g. a large spur becomes High severity).
- **Instant PASS/FAIL Verdict:** Displayed to the operator in < 600 ms, decoupling rapid line signaling from text generation.

### 2.5 On-Device Language Model Advisory
When defects are found, structured findings are passed to an on-device language model:
- **Runtime:** Meta ExecuTorch leveraging the XNNPACK acceleration library optimized for Arm NEON vector instructions.
- **Model:** Llama 3.2 1B Instruct quantized with INT4 group-wise weight lookup tables.
- **Prompt Grounding:** Few-shot examples and Standard Operating Procedure (SOP) tables restrict the model to standard manufacturing actions. The output is forced to begin with `Action:` to eliminate conversational filler.
- **Automated Validation Guardrail:** A post-inference regex engine parses the output text. It verifies that every detected defect and location is referenced, blocks mentions of unobserved defect types, and confirms sentence termination. If any rule is violated or if ExecuTorch is unavailable, the system automatically falls back to deterministic rule-generated instructions.

---

## 3. Results and Empirical Evaluation

### 3.1 Defect Detection Accuracy
The fine-tuned YOLOv8n detector was trained for 25 epochs using Apple Silicon MPS acceleration and exported to ONNX for CPU execution:
- **Overall mAP@50:** 0.942 across all 6 classes.
- **High-Severity Recall:** `missing_hole` (0.981), `open_circuit` (0.965), `short` (0.948).
- **Micro-Defect Recall:** `mouse_bite` (0.892), `spur` (0.916), `spurious_copper` (0.938).

| Defect Class | Base Severity | Precision (@0.30 Conf) | Recall | Standard Remediation Action |
|:---|:---:|:---:|:---:|:---|
| **missing_hole** | High | 0.96 | 0.98 | Cross-reference drill file; re-drill if annular ring permits, else scrap |
| **open_circuit** | High | 0.95 | 0.97 | Continuity test; micro-solder jumper trace if approved, else scrap |
| **short** | High | 0.94 | 0.95 | Cut bridge under magnification; re-verify electrical isolation |
| **mouse_bite** | Medium | 0.89 | 0.89 | Measure remaining trace width against design minimum; repair or reject |
| **spur** | Medium | 0.91 | 0.92 | Verify spacing to adjacent track; trim protrusion if out of tolerance |
| **spurious_copper** | Low | 0.93 | 0.94 | Scrape excess copper if near conductive features; record and pass |

### 3.2 Latency, Throughput, and Telemetry Benchmarks

| Metric | Raspberry Pi 5 (Quad Cortex-A76 @ 2.4 GHz) | Apple M2 Air (Benchmark Baseline) |
|:---|:---:|:---:|
| **Gating Inference (`pcb_gate.onnx`)** | 18.2 ms | 3.4 ms |
| **YOLO Defect Latency (per 640x640 tile)** | 84.5 ms | 11.2 ms |
| **Full Inspection Time (6 tiles)** | **~510 ms** | **~68 ms** |
| **Instant PASS/FAIL Verdict Display** | **< 550 ms** | **< 75 ms** |
| **Llama 3.2 1B TTFT (Time to First Token)** | 480 ms | 95 ms |
| **Llama 3.2 1B Generation Speed** | **11.4 tokens / sec** | **34.8 tokens / sec** |
| **Full Advisory Generation (~35 tokens)** | ~3.1 s | ~1.0 s |
| **Memory Consumption (ExecuTorch RSS)** | **880 MB** | 890 MB |
| **Memory Consumption (Vision + UI RSS)** | **280 MB** | 290 MB |
| **Steady-State Thermal Reading** | **52.4 °C** (Active Cooler) | 41.0 °C |

---

## 4. Hardware Utilization and Optimization Techniques

1. **Two-Stage Decoupled Execution:**
   Separating the vision pipeline and rule-based verdict from LLM inference ensures the operator receives pass/fail sorting commands in ~0.5 seconds. The conversational advisory generates in the background, preventing line bottlenecks.

2. **ExecuTorch INT4 Quantization and XNNPACK:**
   Unquantized Llama 3.2 1B requires over 3.5 GB of RAM. Compiling to an ExecuTorch `.pte` binary with INT4 weight quantization and custom quantized lookup table operators dropped the memory footprint to 880 MB, enabling reliable execution within the memory envelope of a standard edge device.

3. **Sliding-Window Stride Optimization:**
   Selecting a 512 px stride on 640 px tiles provided 128 px of overlap (20% boundary margin). This reduced the required tile count by 40% compared to smaller strides while completely preventing border clipping of defects.

4. **Zero-Overhead Deterministic Fallback:**
   If the language model encounters high load or memory pressure, the system seamlessly outputs rule-based instructions in < 1 ms with zero downtime.

5. **Hardware Telemetry Integration:**
   `hardware_logger.py` samples CPU load, virtual memory percentage, and direct thermal zone sensor data (`/sys/class/thermal/thermal_zone0/temp`), warning operators of potential hardware throttling.

---

## 5. Conclusion and Future Work

This project demonstrates an effective, complete on-device visual inspection station built with edge AI on affordable hardware. By coupling high-resolution tiled vision with on-device language reasoning via ExecuTorch, the system delivers real-time defect localization and actionable rework advice with zero cloud dependency.

**Future Work:**
- **NPU Acceleration:** Compiling YOLOv8 models for the Raspberry Pi AI HAT+ (Hailo-8L NPU) to achieve sub-30 ms full-board inspection.
- **Direct Hardware Actuation:** Connecting Raspberry Pi GPIO pins to pneumatic sorting gates for automated physical board rejection.
- **IPC Standard Fine-Tuning:** Fine-tuning the LLM on IPC-A-600 PCB acceptance standards for deeper compliance reporting.

