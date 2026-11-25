# On-Device PCB Defect Detection and Quality Control Advisory

An edge AI inspection system for manufactured printed circuit boards (PCBs), developed for the Bharat AI Hackathon.

The system runs entirely on-device (target: Raspberry Pi 5) without internet or cloud access. It captures an image of a PCB, identifies surface defects, assigns severity and location, gives the operator clear rework instructions, and logs inspection results.

## How It Works

The inspection pipeline has two stages:

1. **Defect Detection (YOLOv8n):**
   - High-resolution PCB images are cut into overlapping 640x640 tiles (with a 512 px stride) so tiny defects are not lost by downscaling.
   - The YOLO model detects defects on each tile.
   - Duplicate detections across overlapping tile boundaries are merged back into full-image coordinates.
   - The detected defects are: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, and `spurious_copper`.

2. **Operator Advisory (Rules + Llama 3.2 via ExecuTorch):**
   - A deterministic rule engine classifies the severity (High, Medium, Low) and board location (e.g. top-left, centre) of each defect.
   - The operator immediately sees the PASS/FAIL verdict and defect count.
   - Llama 3.2 1B Instruct runs on-device using ExecuTorch (with the XNNPACK backend on Arm CPU) to generate a concise, grounded rework instruction for the operator.
   - The output is validated to make sure every defect and location is covered without hallucinations. If the LLM is unavailable or fails validation, a deterministic rule-based instruction is provided as fallback.
   - Every inspection is saved to a local SQLite database (`inspection_logs.db`), along with CPU load, RAM usage, and device temperature readings.

## Repository Structure

- `dashboard.py` - Streamlit web interface for the operator. Supports live camera capture or image upload, displays detection overlays, advisory text, and hardware telemetry.
- `vision.py` - Core vision pipeline. Handles camera capture, image tiling, YOLO inference, and bounding box merging.
- `llm_engine.py` - Severity rules, location assignment, prompt construction, ExecuTorch runner interface, and response validation.
- `llama_runner.py` - Standalone ExecuTorch runner script for Llama 3.2 1B Instruct.
- `hardware_logger.py` - SQLite logger for inspection history and hardware telemetry (CPU, RAM, core temperature).
- `pipeline.py` - Minimal command-line test script for single image inspection.
- `train_yolo.py` - YOLO training and ONNX export script.
- `download_dataset.py` - Dataset download script using Roboflow.
- `pcb_gate/` - Binary classifier (pcb vs not_pcb) used as an input check before running defect inspection.
- `PCB-Defects-Detection-1/` - Peking University PCB dataset files, including sample test images.
- `PROJECT_REPORT.md` - Final project report detailing methodology, hardware utilization, and results.

## Setup and Installation

### Prerequisites

- Python 3.10+ (tested on Apple Silicon and Raspberry Pi 5)
- Standard USB webcam or camera module

### Installation

```bash
git clone https://github.com/Satyakam-Mishra/BharatAIHackathon.git
cd BharatAIHackathon

python3 -m venv pcb_env
source pcb_env/bin/activate

pip install -r requirements.txt
pip install streamlit pandas psutil
```

## Running the Application

### 1. Operator Dashboard

To start the inspection dashboard:

```bash
streamlit run dashboard.py
```

Open the local URL shown in your terminal (usually http://localhost:8501). From there you can:
- Select input source: camera or uploaded image (sample test images are available in `PCB-Defects-Detection-1/test/images/`).
- Adjust zoom if the camera is positioned far from the board.
- View PASS/FAIL status, annotated boxes, severity table, operator advice, and hardware metrics.

### 2. Standalone Tests

- Test the rule engine and advisory output directly:
  ```bash
  python3 llm_engine.py
  ```

- Test the ExecuTorch Llama runner (requires the ExecuTorch environment on the Pi):
  ```bash
  ~/et_env/bin/python llama_runner.py --test
  ```

- Test single-image vision detection:
  ```bash
  python3 pipeline.py
  ```

## Hardware and Performance Summary

- **Target Device:** Raspberry Pi 5 (Quad-Core Cortex-A76 @ 2.4 GHz)
- **Vision Latency:** ~85 ms per tile (~500 ms for a typical 6-tile full board inspection)
- **Verdict Speed:** Operator receives PASS/FAIL verdict in under 600 ms
- **LLM Speed:** ~11 tokens/second on Pi 5 CPU via ExecuTorch
- **Memory Usage:** ~880 MB for the quantized INT4 Llama model, ~280 MB for the vision and UI processes
- **Operating Mode:** Fully offline, zero cloud reliance

## Dataset

Peking University (PKU) Market PCB Defect Dataset (via Roboflow).

## Contributors

- Satyakam Mishra
- Aryan Lahoti
- Kavya Soni

**Project Guide:** Adithya Kumar
