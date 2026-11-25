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
_focus = None               # result of the last autofocus, see focus.autofocus()


def _open_ribbon_camera():
    """Opens the ribbon camera once, focuses it, and keeps it running. Returns None if there is none."""
    global _picam
    if _picam is not None:
        return _picam
    try:
        from picamera2 import Picamera2      # installed with Raspberry Pi OS
    except ImportError:
        return None
    if not Picamera2.global_camera_info():  # no ribbon camera detected
        return None
    import focus
    cam = Picamera2()
    # Full sensor resolution (4056 x 3040 for the IMX477) for inspection pictures.
    # "RGB888" in Picamera2 stores each pixel as blue, green, red, which is the
    # order OpenCV expects. The small "lores" stream is only used for autofocus.
    config = cam.create_still_configuration(
        main={"size": cam.sensor_resolution, "format": "RGB888"},
        lores={"size": focus.LORES_SIZE},
    )
    cam.configure(config)
    cam.start()
    time.sleep(CSI_WARMUP_S)
    _picam = cam
    refocus()
    return cam


def refocus():
    """Runs autofocus on the ribbon camera and remembers the result.

    The focus motor only holds its position while the camera is running, so this
    runs every time the camera starts. If autofocus cannot run (no focus motor,
    I2C switched off), the last saved setting is used instead. A failure is
    reported but never stops the camera. Returns the result, or None.
    """
    global _focus
    if _picam is None:
        return None
    import focus
    try:
        _focus = focus.autofocus(_picam, focus.find_camera_bus())
        focus.save_focus(_focus["value"])
    except Exception as exc:
        print(f"Autofocus failed: {exc}")
        _focus = None
        try:
            value = focus.saved_focus()
            if value is not None:
                focus.set_focus(value)
        except Exception as exc2:
            print(f"Could not set the saved focus either: {exc2}")
    return _focus


def focus_status():
    """A short description of the current focus, for the dashboard."""
    if _picam is None:
        return None
    if _focus is None:
        return "autofocus not available"
    import focus
    return focus.describe(_focus)


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
