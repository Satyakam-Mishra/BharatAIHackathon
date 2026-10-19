import cv2
import json
import time
from ultralytics import YOLO

# 1. Load Light Vision Model
model = YOLO("yolov8n.onnx", task="detect")

def format_llama_prompt(defect_json):
    """Formats defect metadata into a structured prompt for Llama 3.2 ExecuTorch runtime."""
    system_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are an AI Quality Control Assistant on a factory manufacturing line. "
        "Analyze the provided component inspection JSON and give: "
        "1) Brief Severity Assessment, 2) Actionable Rework Instruction for the operator.<|eot_id|>"
    )
    user_prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Inspection Data: {json.dumps(defect_json, indent=2)}\n<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return system_prompt + user_prompt

def process_inspection_frame(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error loading image: {image_path}")
        return

    # Run YOLO Vision Detection
    start_time = time.time()
    results = model(frame)
    vision_latency_ms = (time.time() - start_time) * 1000

    defect_payload = []
    
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])
            
            # Filter detections by confidence
            if conf > 0.35:
                x1, y1, x2, y2 = [round(val, 1) for val in box.xyxy[0].tolist()]
                defect_payload.append({
                    "defect_class": label,
                    "confidence": round(conf, 2),
                    "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

    print(f"\n--- Vision Pipeline Completed ({vision_latency_ms:.1f} ms) ---")
    
    if defect_payload:
        prompt = format_llama_prompt(defect_payload)
        print("\n Generated LLM Prompt for ExecuTorch:")
        print("="*60)
        print(prompt)
        print("="*60)
    else:
        print("No defects or targets detected. Component nominal.")

if __name__ == "__main__":
    # Test on any image in your dataset folder
    sample_img = "PCB-Defects-Detection-1/test/images/"
    import os
    if os.path.exists(sample_img):
        files = [os.path.join(sample_img, f) for f in os.listdir(sample_img) if f.endswith(('.jpg', '.png'))]
        if files:
            process_inspection_frame(files[0])