"""Camera input for the inspection station.

Two kinds of camera are supported:

  1. A ribbon-cable (CSI) camera on the Raspberry Pi, such as the Arducam IMX477.
     These are read with Picamera2, the Raspberry Pi's own camera library.
     OpenCV cannot open them on a Pi 5.
  2. A USB webcam, such as the Logitech Brio 100, read with OpenCV.

capture_frame() uses the ribbon camera if one is connected, otherwise the USB
webcam. Every picture is returned in OpenCV's BGR colour order.
"""
import time

import cv2

USB_CAMERA_INDEX = 0        # 0 = first USB camera
CSI_WARMUP_S = 2.0          # time for automatic exposure and white balance to settle

_picam = None               # the ribbon camera, kept open between captures (opening takes ~2 s)


def _open_ribbon_camera():
    """Opens the ribbon camera once and keeps it running. Returns None if there is none."""
    global _picam
    if _picam is not None:
        return _picam
    try:
        from picamera2 import Picamera2      # installed with Raspberry Pi OS
    except ImportError:
        return None
    if not Picamera2.global_camera_info():  # no ribbon camera detected
        return None
    cam = Picamera2()
    # Full sensor resolution (4056 x 3040 for the IMX477). "RGB888" in Picamera2
    # stores each pixel as blue, green, red, which is the order OpenCV expects.
    config = cam.create_still_configuration(main={"size": cam.sensor_resolution, "format": "RGB888"})
    cam.configure(config)
    cam.start()
    time.sleep(CSI_WARMUP_S)
    _apply_saved_focus()
    _picam = cam
    return cam


def _apply_saved_focus():
    """Moves the motorised lens to the setting saved by focus.py, if there is one.

    The focus motor only holds its position while the camera is running, so this
    is done every time the camera starts. A failure here is reported but does not
    stop the camera.
    """
    try:
        import focus
        value = focus.saved_focus()
        if value is not None:
            focus.set_focus(value)
            time.sleep(focus.SETTLE_S)
    except Exception as exc:
        print(f"Could not set the lens focus: {exc}")


def camera_name():
    """A short description of the camera that capture_frame() will use."""
    cam = _open_ribbon_camera()
    if cam is not None:
        model = cam.camera_properties.get("Model", "ribbon camera")
        width, height = cam.sensor_resolution
        return f"{model} ribbon camera ({width} x {height})"
    return f"USB camera {USB_CAMERA_INDEX}"


def capture_frame():
    """Takes one picture. Returns it as a BGR array, or None if no camera works."""
    cam = _open_ribbon_camera()
    if cam is not None:
        return cam.capture_array("main")

    cap = cv2.VideoCapture(USB_CAMERA_INDEX)
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
