import streamlit as st
import cv2
import numpy as np
import sqlite3
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from hardware_logger import init_db, get_hardware_metrics, log_inspection

st.set_page_config(page_title="Edge AI PCB Quality Control", layout="wide")
init_db()

@st.cache_resource
def load_vision_model():
    return YOLO("yolov8n.onnx", task="detect")

model = load_vision_model()

st.title("🔧 On-Device PCB Visual Inspection Assistant")

# Top Hardware Telemetry Bar
hw = get_hardware_metrics()
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("RPi 5 CPU Load", f"{hw['cpu_percent']}%")
col_m2.metric("RAM Usage", f"{hw['ram_percent']}%")
col_m3.metric("Core Temperature", f"{hw['temp_c']} °C")

st.divider()

# Main Layout: Left Vision Feed | Right Advisory & Logs
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📸 Live Camera / PCB Input")
    input_mode = st.radio("Select Input Source:", ["Upload Test PCB Image", "Live Camera Capture"], horizontal=True)
    
    img_file = None
    if input_mode == "Live Camera Capture":
        img_file = st.camera_input("Capture PCB on Inspection Rig")
    else:
        img_file = st.file_uploader("Upload PCB Image", type=["jpg", "png", "jpeg"])

    if img_file is not None:
        image = Image.open(img_file)
        frame = np.array(image)
        
        # Run YOLO Inference
        results = model(frame)
        annotated_frame = results[0].plot()
        st.image(annotated_frame, caption="YOLO Defect Detection Overlay", use_container_width=True)
        
        # Extract metadata
        detections = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.30:
                cls_id = int(box.cls[0])
                detections.append({
                    "defect": model.names[cls_id],
                    "confidence": round(conf, 2)
                })
                
        # Trigger Llama Advisory (Placeholder until CSE links .pte binary)
        if detections:
            advice = f"⚠️ Found {len(detections)} issue(s) ({detections[0]['defect']}). Inspect trace continuity and clear copper bridges before soldering."
        else:
            advice = "✅ Component Nominal. Proceed to next assembly stage."
            
        log_inspection(detections, advice)
        st.session_state["last_advice"] = advice
        st.session_state["last_detections"] = detections

with right_col:
    st.subheader("🤖 Llama 3.2 Edge Advisory (ExecuTorch)")
    if "last_advice" in st.session_state:
        st.info(st.session_state["last_advice"])
        st.json(st.session_state.get("last_detections", []))
    else:
        st.write("Awaiting PCB scan...")

    st.subheader("📊 Recent Factory Inspection Logs")
    conn = sqlite3.connect("inspection_logs.db")
    df = pd.read_sql_query("SELECT * FROM inspections ORDER BY id DESC LIMIT 8", conn)
    conn.close()
    st.dataframe(df, use_container_width=True)