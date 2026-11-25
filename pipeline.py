"""Inspects one picture from the command line, without the dashboard.

Runs the same steps as the dashboard: PCB check, tiled defect search, severity and
location rules, then the Llama advisory (or the rule-based instruction if Llama is not
available). The picture with boxes drawn on it is saved next to the original.

    python3 pipeline.py path/to/board.jpg
"""
import os
import sys

import cv2

import llm_engine
import vision


def inspect(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read the picture: {image_path}")
        return

    gate = vision.load_gate()
    if gate is not None:
        prob = vision.pcb_probability(gate, frame)
        print(f"PCB check: p(PCB) = {prob:.2f}")
        if prob < vision.GATE_THRESHOLD:
            print("Result: NO PCB IN VIEW (neither PASS nor FAIL)")
            return
    else:
        print("PCB check: pcb_gate.onnx not found, check skipped")

    model, model_path = vision.load_model()
    found = vision.detect(model, frame, use_tiles=(model_path == vision.PCB_MODEL))
    height, width = frame.shape[:2]
    findings = llm_engine.add_severity_and_location(found["detections"], width, height)
    print(f"Defect search: {found['tiles']} tiles in {found['vision_ms']:.0f} ms")
    print("Result:", "FAIL" if findings else "PASS")

    advice = llm_engine.advise(findings)
    print(advice["summary"])
    print(advice["text"], f"[written by: {advice['source']}]")

    out_path = os.path.splitext(image_path)[0] + "_inspected.jpg"
    cv2.imwrite(out_path, found["annotated_bgr"])
    print("Picture with boxes saved to:", out_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 pipeline.py path/to/board.jpg")
    else:
        inspect(sys.argv[1])
