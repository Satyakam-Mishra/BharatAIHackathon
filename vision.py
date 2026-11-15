"""Vision pipeline: YOLOv8n defect detector, run with ONNX Runtime on the CPU.

The PCB model was trained on 640 x 640 tiles cut from full-size photos, so a
photo is inspected the same way here: cut it into overlapping tiles, run the
model on each tile, shift the boxes back to their place in the full photo and
merge the duplicates found where tiles overlap.

This file has nothing to do with Llama. The advisory pipeline lives in
llm_engine.py and runs on a different runtime (ExecuTorch).
"""
import os
import time

import cv2
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
PCB_MODEL = "yolo_pcb.onnx"       # our fine-tuned PCB defect model (from Colab)
FALLBACK_MODEL = "yolov8n.onnx"   # stock model, only for testing the plumbing
TILE = 640                        # tile size in pixels, same as in training
STRIDE = 512                      # step between tiles (640 - 512 = 128 px overlap)
CONF_THRESHOLD = 0.30             # ignore detections below 30% confidence
CAMERA_INDEX = 0                  # 0 = first camera plugged into this computer

# One colour per class (BGR order, because OpenCV draws in BGR)
COLOURS = [(0, 0, 255), (0, 165, 255), (255, 0, 255), (255, 0, 0), (0, 200, 0), (200, 200, 0)]


def load_model():
    """Loads the PCB model if its file exists, otherwise the stock test model."""
    path = PCB_MODEL if os.path.exists(PCB_MODEL) else FALLBACK_MODEL
    return YOLO(path, task="detect"), path


def capture_frame():
    """Takes one picture from the USB camera of the computer running this code.

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


def tile_starts(length):
    """Where each tile begins along one side, so the tiles cover it completely."""
    if length <= TILE:
        return [0]
    starts = list(range(0, length - TILE + 1, STRIDE))
    if starts[-1] != length - TILE:
        starts.append(length - TILE)      # one last tile flush with the far edge
    return starts


def merge_duplicates(detections):
    """Merges boxes that describe the same defect seen in two overlapping tiles.

    Two boxes count as the same defect when their overlap covers more than half
    of the smaller box. The merged box is the outline around both, and it keeps
    the higher confidence.
    """
    merged = []
    for det in sorted(detections, key=lambda d: d["confidence"], reverse=True):
        x1, y1, x2, y2 = det["box"]
        for kept in merged:
            kx1, ky1, kx2, ky2 = kept["box"]
            overlap_w = min(x2, kx2) - max(x1, kx1)
            overlap_h = min(y2, ky2) - max(y1, ky1)
            if overlap_w <= 0 or overlap_h <= 0:
                continue
            smaller = min((x2 - x1) * (y2 - y1), (kx2 - kx1) * (ky2 - ky1))
            if overlap_w * overlap_h > 0.5 * smaller:
                kept["box"] = [min(x1, kx1), min(y1, ky1), max(x2, kx2), max(y2, ky2)]
                break
        else:
            merged.append(dict(det))
    return merged


def draw_detections(frame_bgr, detections, class_names):
    """Returns a copy of the picture with a labelled box drawn around each defect."""
    out = frame_bgr.copy()
    longest = max(out.shape[:2])
    thickness = max(2, round(longest / 600))
    font_scale = max(0.6, longest / 1500)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        colour = COLOURS[class_names.index(det["defect"]) % len(COLOURS)] if det["defect"] in class_names else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        label = f"{det['defect']} {det['confidence']:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, thickness)
    return out


def detect(model, frame_bgr, use_tiles=True, zoom=1.0):
    """Finds defects in one picture.

    model      : the YOLO model from load_model()
    frame_bgr  : the picture as an OpenCV array in BGR colour order
    use_tiles  : cut the picture into tiles (needed for the PCB model)
    zoom       : enlarge the picture first. Useful when the camera is far from the
                 board, so that defects appear about as large as in the training tiles.

    Returns a dictionary with the list of detections (box coordinates refer to the
    original picture), the annotated picture, the time taken and the tile count.
    """
    t0 = time.perf_counter()
    work = frame_bgr
    if zoom != 1.0:
        work = cv2.resize(frame_bgr, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    height, width = work.shape[:2]

    if use_tiles:
        tiles = [(x0, y0) for y0 in tile_starts(height) for x0 in tile_starts(width)]
    else:
        tiles = [(0, 0)]

    names = model.names
    detections = []
    for x0, y0 in tiles:
        piece = work[y0:y0 + TILE, x0:x0 + TILE] if use_tiles else work
        result = model(piece, conf=CONF_THRESHOLD, imgsz=TILE, verbose=False)[0]
        for box in result.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            detections.append({
                "defect": names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 2),
                # shift from tile coordinates to full-picture coordinates, and undo the zoom
                "box": [int((bx1 + x0) / zoom), int((by1 + y0) / zoom),
                        int((bx2 + x0) / zoom), int((by2 + y0) / zoom)],
            })

    detections = merge_duplicates(detections)
    detections.sort(key=lambda d: (d["box"][1], d["box"][0]))    # top-to-bottom, left-to-right
    vision_ms = (time.perf_counter() - t0) * 1000

    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    return {
        "detections": detections,
        "annotated_bgr": draw_detections(frame_bgr, detections, class_names),
        "vision_ms": vision_ms,
        "tiles": len(tiles),
    }
