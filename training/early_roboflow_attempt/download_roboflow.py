"""Early attempt (not used for the final model): downloads the Roboflow version of the PKU dataset.

The Roboflow API key is read from the ROBOFLOW_API_KEY environment variable, so it never goes into Git:
    export ROBOFLOW_API_KEY=your_key_here
    python3 download_roboflow.py
"""
import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
# Download PKU PCB dataset formatted for YOLOv8/v11
project = rf.workspace("jr-mqqnk").project("pcb-defects-detection-anddl")
dataset = project.version(1).download("yolov8")

print(f"Dataset downloaded to: {dataset.location}")