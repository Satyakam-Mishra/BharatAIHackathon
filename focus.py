"""Focus control for the Arducam IMX477 motorised-focus camera on a Raspberry Pi 5.

The lens is moved by a small focus motor that listens on the camera's own I2C
bus at address 0x0C. The Pi's camera software does not drive it, so this file
does. The motor takes a number from 0 to 1023; which number gives a sharp
picture depends on the distance to the board, so we find it by trying values
and measuring sharpness.

The motor only holds its position while the camera is running, so the focus is
set again every time the camera is started (camera.py does this automatically).

Usage, on the Pi, with the dashboard STOPPED (only one program can use the camera):

    python3 focus.py sweep        # find the sharpest setting and save it
    python3 focus.py set 450      # save a setting by hand
    python3 focus.py photo        # take focus_test.jpg with the saved setting
"""
import glob
import os
import subprocess
import sys
import time

import cv2

FOCUS_I2C_ADDRESS = 0x0C          # address of the focus motor chip
SENSOR_I2C_ADDRESS = 0x1A         # address of the IMX477 image sensor, used to find the right bus
FOCUS_MIN, FOCUS_MAX = 0, 1023    # the motor takes a 10-bit number
SETTLE_S = 0.25                   # time for the lens to stop moving after a change
FOCUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus_value.txt")


def find_camera_bus():
    """Returns the number of the I2C bus the camera is on.

    When the IMX477 driver is loaded, Linux lists the sensor as "<bus>-001a",
    so the number before the dash is the camera's bus.
    """
    matches = sorted(glob.glob(f"/sys/bus/i2c/devices/*-{SENSOR_I2C_ADDRESS:04x}"))
    if not matches:
        raise RuntimeError("No IMX477 found. Check the cable and the dtoverlay=imx477 line in config.txt.")
    bus = int(os.path.basename(matches[0]).split("-")[0])
    if not os.path.exists(f"/dev/i2c-{bus}"):
        raise RuntimeError(f"/dev/i2c-{bus} is missing. Turn I2C on with: sudo raspi-config nonint do_i2c 0")
    return bus


def set_focus(value, bus=None):
    """Moves the lens to a focus setting from 0 to 1023.

    The 10-bit value is split over two bytes in the way Arducam's own example
    code does: the top 6 bits go in the first byte, the bottom 4 bits in the
    top half of the second byte.
    """
    value = max(FOCUS_MIN, min(FOCUS_MAX, int(value)))
    bus = find_camera_bus() if bus is None else bus
    shifted = (value << 4) & 0x3FF0
    byte1 = (shifted >> 8) & 0x3F
    byte2 = shifted & 0xF0
    subprocess.run(["i2cset", "-y", str(bus), hex(FOCUS_I2C_ADDRESS), str(byte1), str(byte2)],
                   check=True, capture_output=True, text=True)
    return value


def saved_focus():
    """The focus setting saved by 'sweep' or 'set', or None if there is none."""
    try:
        with open(FOCUS_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def save_focus(value):
    with open(FOCUS_FILE, "w") as f:
        f.write(str(int(value)))


def sharpness(frame_bgr):
    """How sharp the middle of the picture is (higher is sharper).

    The Laplacian highlights edges; a blurry picture has weak edges, so the
    variance of the Laplacian is low. Only the central half of the picture is
    used, and it is shrunk first to keep this fast.
    """
    h, w = frame_bgr.shape[:2]
    centre = frame_bgr[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    small = cv2.resize(centre, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _open_camera():
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": cam.sensor_resolution, "format": "RGB888"}))
    cam.start()
    time.sleep(2)                 # let exposure and white balance settle
    return cam


def _frame_at(cam, value, bus):
    """Moves the lens, waits, and returns a fresh frame taken at that setting."""
    set_focus(value, bus)
    time.sleep(SETTLE_S)
    cam.capture_array("main")     # throw away a frame that may have been exposed while the lens moved
    return cam.capture_array("main")


def sweep():
    """Tries focus settings from near to far and saves the sharpest one."""
    bus = find_camera_bus()
    cam = _open_camera()
    try:
        print("Coarse sweep (put a printed PCB sheet under the camera now):")
        scores = {}
        for value in range(FOCUS_MIN, FOCUS_MAX + 1, 50):
            scores[value] = sharpness(_frame_at(cam, value, bus))
            print(f"  focus {value:4d}: sharpness {scores[value]:9.1f}")
        best = max(scores, key=scores.get)

        print("Fine sweep around the best setting:")
        for value in range(max(FOCUS_MIN, best - 50), min(FOCUS_MAX, best + 50) + 1, 10):
            if value not in scores:
                scores[value] = sharpness(_frame_at(cam, value, bus))
                print(f"  focus {value:4d}: sharpness {scores[value]:9.1f}")
        best = max(scores, key=scores.get)

        frame = _frame_at(cam, best, bus)
        cv2.imwrite("focus_test.jpg", frame)
    finally:
        cam.stop()
        cam.close()
    save_focus(best)
    print(f"\nSharpest setting: {best} (saved to {FOCUS_FILE}). Photo saved as focus_test.jpg.")


def photo():
    """Takes focus_test.jpg using the saved focus setting."""
    value = saved_focus()
    if value is None:
        sys.exit("No saved focus setting. Run: python3 focus.py sweep")
    bus = find_camera_bus()
    cam = _open_camera()
    try:
        frame = _frame_at(cam, value, bus)
    finally:
        cam.stop()
        cam.close()
    cv2.imwrite("focus_test.jpg", frame)
    print(f"focus_test.jpg taken at focus {value} (sharpness {sharpness(frame):.1f}).")


def main():
    args = sys.argv[1:]
    if args == ["sweep"]:
        sweep()
    elif args == ["photo"]:
        photo()
    elif len(args) == 2 and args[0] == "set" and args[1].isdigit():
        value = max(FOCUS_MIN, min(FOCUS_MAX, int(args[1])))
        save_focus(value)
        print(f"Saved focus setting {value}. Run 'python3 focus.py photo' to check it.")
    else:
        sys.exit("Usage: python3 focus.py sweep | set NUMBER | photo")


if __name__ == "__main__":
    main()
