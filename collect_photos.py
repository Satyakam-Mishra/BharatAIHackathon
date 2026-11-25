"""Takes a series of photos with the inspection camera, for training the PCB check.

Usage, on the Raspberry Pi with the dashboard's environment active:

    python3 collect_photos.py pcb 40        # 40 photos that DO show a PCB
    python3 collect_photos.py not_pcb 60    # 60 photos that do NOT show a PCB

A photo is taken every few seconds. Between shots, change what is under the camera
(move, rotate or swap the board or object), so that no two photos are alike.
Photos are saved to ~/gate_photos/<label>/, shrunk to at most 1280 pixels wide.
"""
import os
import sys
import time

import cv2

import camera

SECONDS_BETWEEN_SHOTS = 4
MAX_SIDE = 1280
LABELS = ("pcb", "not_pcb")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in LABELS or not sys.argv[2].isdigit():
        sys.exit("Usage: python3 collect_photos.py pcb|not_pcb NUMBER_OF_PHOTOS")
    label, count = sys.argv[1], int(sys.argv[2])

    folder = os.path.expanduser(f"~/gate_photos/{label}")
    os.makedirs(folder, exist_ok=True)
    print(f"Camera: {camera.camera_name()}")
    print(f"Saving {count} '{label}' photos to {folder}. Press Ctrl + C to stop early.\n")

    for i in range(1, count + 1):
        for remaining in range(SECONDS_BETWEEN_SHOTS, 0, -1):
            print(f"\r  photo {i}/{count} in {remaining} s ... change the scene now   ", end="", flush=True)
            time.sleep(1)
        frame = camera.capture_frame()
        if frame is None:
            sys.exit("\nCould not read from the camera.")
        scale = MAX_SIDE / max(frame.shape[:2])
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        path = os.path.join(folder, f"{label}_{time.strftime('%Y%m%d_%H%M%S')}_{i:03d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\r  saved {os.path.basename(path)}" + " " * 30)

    print(f"\nDone. {len(os.listdir(folder))} '{label}' photos in {folder} so far.")


if __name__ == "__main__":
    main()
