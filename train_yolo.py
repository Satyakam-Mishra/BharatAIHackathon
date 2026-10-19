from ultralytics import YOLO

# 1. Load pretrained nano model
model = YOLO("yolov8n.pt")

# 2. Train on M2 Air using Apple Silicon MPS GPU
results = model.train(
    data="PCB-Defects-Detection-1/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    device="mps",
    workers=4,
    project="pcb_inspection",
    name="yolov8n_pcb_v1"
)

# 3. Export to ONNX format for Raspberry Pi 5 CPU execution
onnx_path = model.export(format="onnx", dynamic=False)
print(f"\nModel successfully exported to ONNX: {onnx_path}")