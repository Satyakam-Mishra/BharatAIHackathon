import os
import sqlite3
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from hardware_logger import init_db, get_hardware_metrics, log_inspection
from llm_engine import get_advisory_recommendation

# ---------------------------------------------------------------------------
# Settings (change these in one place)
# ---------------------------------------------------------------------------
PCB_MODEL = "yolo_pcb.onnx"       # our fine-tuned PCB defect model (from Colab)
FALLBACK_MODEL = "yolov8n.onnx"   # stock model, only for testing the pipeline
CONF_THRESHOLD = 0.30             # ignore detections the model is less than 30% sure about
CAMERA_INDEX = 0                  # 0 = first camera plugged into this computer
SOURCE_CAMERA = "Camera on this device"
SOURCE_UPLOAD = "Upload an image"

st.set_page_config(page_title="Edge AI PCB Quality Control", layout="wide")
init_db()


@st.cache_resource
def load_vision_model():
    """Loads the YOLO model once and keeps it in memory between page reruns."""
    path = PCB_MODEL if os.path.exists(PCB_MODEL) else FALLBACK_MODEL
    return YOLO(path, task="detect"), path


def capture_frame():
    """Takes one picture from the USB camera of the computer running this app.

    On the Raspberry Pi this is the webcam plugged into the Pi, not the camera
    of the laptop that is only showing the web page.
    Returns the picture in OpenCV's BGR colour order, or None if it failed.
    """
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # The first few frames are often dark while the camera adjusts its exposure,
    # so we read and throw away five frames before keeping one.
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def run_inspection(frame_bgr):
    """Runs one full inspection: vision -> advisory -> log. Stores the result for display."""
    # 1. Vision: find the defects
    t0 = time.perf_counter()
    results = model(frame_bgr, conf=CONF_THRESHOLD, verbose=False)
    vision_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        detections.append({
            "defect": model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 2),
            "box": [x1, y1, x2, y2],
        })

    # 2. Advisory: turn the findings into operator advice
    t1 = time.perf_counter()
    advice = get_advisory_recommendation(detections)
    advice_ms = (time.perf_counter() - t1) * 1000

    # 3. Log: one row per inspection
    log_inspection(detections, advice)

    # Keep the result so it stays on screen when Streamlit reruns the script.
    # plot() draws the boxes and returns BGR, so we convert to RGB for the browser.
    st.session_state["last_result"] = {
        "image_rgb": cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB),
        "detections": detections,
        "advice": advice,
        "vision_ms": vision_ms,
        "advice_ms": advice_ms,
    }


@st.fragment(run_every="2s")
def show_telemetry():
    """Hardware readings. Only this small section reruns every 2 seconds."""
    hw = get_hardware_metrics()
    c1, c2, c3 = st.columns(3)
    c1.metric("CPU Load", f"{hw['cpu_percent']}%")
    c2.metric("RAM Usage", f"{hw['ram_percent']}%")
    c3.metric("Core Temperature", f"{hw['temp_c']} °C")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
model, model_path = load_vision_model()

st.title("🔧 On-Device PCB Visual Inspection Assistant")
show_telemetry()
if model_path == FALLBACK_MODEL:
    st.warning(
        f"'{PCB_MODEL}' was not found, so the stock test model is loaded. "
        "It recognises everyday objects, not PCB defects."
    )
st.divider()

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📸 PCB Input")
    source = st.radio("Select input source:", [SOURCE_CAMERA, SOURCE_UPLOAD], horizontal=True)

    # new_frame stays None unless a NEW picture arrives in this run.
    # This is what stops the same picture being inspected and logged again
    # every time Streamlit reruns the script.
    new_frame = None

    if source == SOURCE_CAMERA:
        if st.button("Capture and inspect", type="primary"):
            new_frame = capture_frame()
            if new_frame is None:
                st.error("Could not read from the camera. Check that it is plugged in.")
    else:
        uploaded = st.file_uploader("Upload PCB image", type=["jpg", "png", "jpeg"])
        if uploaded is None:
            st.session_state["last_upload_id"] = None
        elif st.session_state.get("last_upload_id") != uploaded.file_id:
            st.session_state["last_upload_id"] = uploaded.file_id
            rgb = np.array(Image.open(uploaded).convert("RGB"))
            new_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # YOLO expects BGR arrays

    if new_frame is not None:
        with st.spinner("Inspecting..."):
            run_inspection(new_frame)

    last = st.session_state.get("last_result")
    if last:
        st.image(last["image_rgb"], caption="Detection overlay")
        st.caption(
            f"Vision: {last['vision_ms']:.0f} ms  |  Advisory: {last['advice_ms']:.0f} ms  |  "
            f"Model: {model_path}"
        )

with right_col:
    st.subheader("🤖 Llama 3.2 Edge Advisory (ExecuTorch)")
    if last:
        if last["detections"]:
            st.error(f"FAIL: {len(last['detections'])} defect(s) found")
        else:
            st.success("PASS: no defects found")
        st.info(last["advice"])
        st.json(last["detections"])
    else:
        st.write("Awaiting PCB scan...")

    st.subheader("📊 Recent Inspection Logs")
    conn = sqlite3.connect("inspection_logs.db")
    df = pd.read_sql_query("SELECT * FROM inspections ORDER BY id DESC LIMIT 8", conn)
    conn.close()
    st.dataframe(df)
