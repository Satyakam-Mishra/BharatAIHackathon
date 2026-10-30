import os
import json
import subprocess

# Paths expected on the Raspberry Pi 5
RUNNER_PATH = "./llama_main"
MODEL_PATH = "llama3_2_1b_xnnpack.pte"
TOKENIZER_PATH = "tokenizer.model"

# Domain-specific PCB rework knowledge for fallback testing on Windows/Mac
PCB_DEFECT_KNOWLEDGE = {
    "missing_hole": "Severity: High (Mechanical/Electrical Failure). Via or component through-hole was unplated or skipped during CNC drilling. Rework: Route board to manual micro-drill station (check Gerber drill file for bit diameter) or scrap if internal multilayer barrel is disconnected.",
    "mouse_bite": "Severity: Medium (Trace Width Reduction). Over-etching or depaneling nibbles along copper edge reduce current capacity. Rework: Measure remaining trace width; if <70% of nominal spec, apply conductive epoxy or bridge with 30-AWG jumper wire.",
    "open_circuit": "Severity: Critical (Signal Break). Complete discontinuity in copper trace caused by photoresist scratch or over-etching. Rework: Scrape solder mask 2mm on both sides of break, tin copper pads, and solder a micro-jumper bridge across the gap.",
    "short": "Severity: Critical (Electrical Short). Unwanted copper bridge connecting adjacent traces due to under-etching. Rework: Use a surgical #11 scalpel blade under a microscope to sever the copper bridge, then verify isolation with a multimeter continuity test.",
    "spur": "Severity: Medium (Clearance Violation). Sharp copper projection extending from trace, risking high-voltage arcing. Rework: Carefully shave off the metallic protrusion using a fine micro-chisel and reseal exposed base laminate with UV-curing solder mask.",
    "spurious_copper": "Severity: Low-Medium (Parasitic Capacitance/Short Risk). Isolated residual copper island left on substrate. Rework: Lift isolated copper speck using a heated rework tip or fine blade if within 0.2mm of active signal pads."
}

def build_llama_prompt(detections):
    """Constructs the official Llama 3.2 Instruct prompt template."""
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are an on-device AI Quality Control Assistant on a PCB manufacturing floor. "
        f"Given detected surface defects, state the severity, root cause, and exact operator rework steps in 2 concise sentences.<|eot_id|>\n"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Detected PCB Defects: {json.dumps(detections)}<|eot_id|>\n"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )

def get_advisory_recommendation(detections):
    """
    Runs on-device Llama 3.2 via ExecuTorch if the binary is present on RPi 5.
    Otherwise falls back to structured PCB rule-based reasoning for local UI testing.
    """
    if not detections:
        return "✅ Status Nominal: No surface defects detected. Advance PCB to automated optical solder paste inspection."

    prompt = build_llama_prompt(detections)

    # 1. Check if running on Raspberry Pi 5 with compiled ExecuTorch artifacts
    if os.path.exists(RUNNER_PATH) and os.path.exists(MODEL_PATH):
        try:
            cmd = [
                RUNNER_PATH,
                "--model_path", MODEL_PATH,
                "--tokenizer_path", TOKENIZER_PATH,
                "--prompt", prompt,
                "--temperature", "0.2",
                "--seq_len", "256"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception as e:
            return f"ExecuTorch Runtime Warning: {e}"

    # 2. Local Development Fallback (Windows / Mac before .pte copy)
    primary_defect = detections[0]["defect"]
    conf = detections[0]["confidence"]
    count = len(detections)

    if primary_defect in PCB_DEFECT_KNOWLEDGE:
        return f"[Defect: {primary_defect.upper()} | Count: {count} | Conf: {conf}] {PCB_DEFECT_KNOWLEDGE[primary_defect]}"
    
    # Generic object detected (when testing with base yolov8n.onnx before fine-tuned PCB weights arrive)
    return (
        f"[Detected: {primary_defect} ({count}x) at {conf*100:.0f}% confidence] "
        f"Severity: Medium. Foreign object or unclassified surface anomaly detected on inspection bed. "
        f"Rework: Clear inspection fixture and verify `yolo_pcb_best.onnx` weights are loaded."
    )