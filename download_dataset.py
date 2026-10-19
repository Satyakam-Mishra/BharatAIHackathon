import os
from roboflow import Roboflow

# Initialize Roboflow download (using public universe dataset)
rf = Roboflow(api_key="LZCoHpL2pJhUgDANEn21") # Or download directly via Ultralytics dataset link
# Download PKU PCB dataset formatted for YOLOv8/v11
project = rf.workspace("jr-mqqnk").project("pcb-defects-detection-anddl")
dataset = project.version(1).download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")