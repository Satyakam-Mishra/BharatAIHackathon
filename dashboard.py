import sqlite3
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import llm_engine
import vision
from hardware_logger import init_db, get_hardware_metrics, log_inspection

SOURCE_CAMERA = "Camera on this device"
SOURCE_UPLOAD = "Upload an image"
DISPLAY_WIDTH = 1400              # shrink very large photos to this width before showing them

st.set_page_config(page_title="Edge AI PCB Quality Control", layout="wide")
init_db()


@st.cache_resource
def load_vision_model():
    """Loads the YOLO model once and keeps it in memory between page reruns."""
    return vision.load_model()


def run_inspection(frame_bgr, zoom):
    """Runs one full inspection: vision -> rules -> Llama advisory -> log. Stores the result for display."""
    # 1. Vision: find the defects (see vision.py)
    found = vision.detect(model, frame_bgr, use_tiles=is_pcb_model, zoom=zoom)

    # 2. Rules: give every defect a severity and a location name (see llm_engine.py)
    height, width = frame_bgr.shape[:2]
    findings = llm_engine.add_severity_and_location(found["detections"], width, height)

    # 3. Advisory: Llama on ExecuTorch turns the findings into a note for the operator
    t0 = time.perf_counter()
    advice = llm_engine.advise(findings)
    advice_ms = (time.perf_counter() - t0) * 1000

    # 4. Log: one row per inspection
    log_inspection(findings, advice["text"])

    # Prepare the picture for the browser: shrink it if it is huge, and convert
    # OpenCV's BGR colour order to the RGB order that browsers expect.
    picture = found["annotated_bgr"]
    if picture.shape[1] > DISPLAY_WIDTH:
        scale = DISPLAY_WIDTH / picture.shape[1]
        picture = cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Keep the result so it stays on screen when Streamlit reruns the script.
    st.session_state["last_result"] = {
        "image_rgb": cv2.cvtColor(picture, cv2.COLOR_BGR2RGB),
        "findings": findings,
        "advice": advice,
        "vision_ms": found["vision_ms"],
        "tiles": found["tiles"],
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
is_pcb_model = model_path == vision.PCB_MODEL

st.title("🔧 On-Device PCB Visual Inspection Assistant")
show_telemetry()
if not is_pcb_model:
    st.warning(
        f"'{vision.PCB_MODEL}' was not found, so the stock test model is loaded. "
        "It recognises everyday objects, not PCB defects."
    )
st.divider()

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("📸 PCB Input")
    source = st.radio("Select input source:", [SOURCE_CAMERA, SOURCE_UPLOAD], horizontal=True)
    zoom = st.slider(
        "Zoom before inspection", 1.0, 3.0, 1.0, 0.5,
        help="Enlarges the picture before it is cut into tiles. Leave at 1.0 for the full-size "
             "dataset photos. Raise it when the camera is far from the board and defects look tiny.",
    )

    # new_frame stays None unless a NEW picture arrives in this run.
    # This is what stops the same picture being inspected and logged again
    # every time Streamlit reruns the script.
    new_frame = None

    if source == SOURCE_CAMERA:
        if st.button("Capture and inspect", type="primary"):
            new_frame = vision.capture_frame()
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
        with st.spinner("Inspecting: vision first, then the Llama advisory..."):
            run_inspection(new_frame, zoom)

    last = st.session_state.get("last_result")
    if last:
        st.image(last["image_rgb"], caption="Detection overlay", output_format="JPEG")
        st.caption(
            f"Vision: {last['vision_ms']:.0f} ms over {last['tiles']} tile(s)  |  "
            f"Advisory: {last['advice_ms']:.0f} ms  |  Model: {model_path}"
        )

with right_col:
    st.subheader("🤖 Operator Advisory")
    if last:
        findings, advice = last["findings"], last["advice"]
        if findings:
            st.error(f"FAIL: {len(findings)} defect(s) found, highest severity {llm_engine.highest_severity(findings)}")
        else:
            st.success("PASS: no defects found")

        # Two spaces before a line break make Markdown keep the three lines separate.
        st.info(advice["text"].replace("\n", "  \n"))

        if advice["source"] == "llama":
            stats = advice["stats"]
            st.caption(
                f"Written on this device by Llama 3.2 1B (ExecuTorch, XNNPACK). "
                f"{stats.get('tokens_per_second', '?')} tokens/s, "
                f"first token after {stats.get('time_to_first_token_ms', '?')} ms, "
                f"peak memory {stats.get('peak_memory_mb', '?')} MB."
            )
        elif advice["source"] == "rules":
            st.caption(f"Written by the built-in rules, not by Llama ({advice.get('error', 'Llama unavailable')}).")

        if findings:
            table = pd.DataFrame(findings)[["defect", "severity", "location", "size_pct", "confidence"]]
            st.dataframe(table, hide_index=True)
    else:
        st.write("Awaiting PCB scan...")

    st.subheader("📊 Recent Inspection Logs")
    conn = sqlite3.connect("inspection_logs.db")
    df = pd.read_sql_query("SELECT * FROM inspections ORDER BY id DESC LIMIT 8", conn)
    conn.close()
    st.dataframe(df)
