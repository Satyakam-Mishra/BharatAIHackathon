"""Automatic focus for the Arducam IMX477 motorised-focus camera on a Raspberry Pi 5.

The lens is moved by a small focus motor that listens on the camera's own I2C
bus at address 0x0C. The Pi's camera software does not drive it, so this file
does. The motor takes a number from 0 to 1023.

Autofocus works like a phone camera's "contrast" autofocus: move the lens to a
series of positions, measure how sharp the picture is at each one, and keep the
sharpest. It uses a small extra picture stream ("lores", 800 x 600) so each
measurement is quick. It takes about 10 seconds. Keep your hands out of view
while it runs.

camera.py runs autofocus automatically when the camera starts, and the
dashboard has a "Refocus camera" button. From the terminal (dashboard stopped,
because only one program can use the camera):

    python3 focus.py tune     # for setting the lens by hand: turn, step back, press Enter
    python3 focus.py sweep    # run autofocus once, save the result, take focus_test.jpg
    python3 focus.py photo    # take focus_test.jpg with the saved setting
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
COARSE_STEP, FINE_STEP = 64, 16   # autofocus tries every 64th setting, then every 16th near the best
SETTLE_S = 0.15                   # time for the lens to stop moving after a change
EDGE_MARGIN = 64                  # a best setting this close to 0 or 1023 means the lens needs turning
LORES_SIZE = (800, 600)           # the small picture stream used for measuring sharpness
FOCUS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus_value.txt")


# ---------------------------------------------------------------------------
# Talking to the focus motor
# ---------------------------------------------------------------------------
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
    """The last focus setting found, or None if there is none."""
    try:
        with open(FOCUS_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def save_focus(value):
    with open(FOCUS_FILE, "w") as f:
        f.write(str(int(value)))


# ---------------------------------------------------------------------------
# Measuring sharpness and searching for the sharpest setting
# ---------------------------------------------------------------------------
def sharpness_gray(gray):
    """How sharp the middle of a grey picture is (higher is sharper).

    The Laplacian highlights edges; a blurry picture has weak edges, so the
    variance of the Laplacian is low. Only the central half is used, because
    that is where the board should be.
    """
    h, w = gray.shape[:2]
    centre = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return float(cv2.Laplacian(centre, cv2.CV_64F).var())


def sharpness(frame_bgr):
    """Same as sharpness_gray, for a colour picture."""
    return sharpness_gray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))


def lores_gray(cam):
    """Grabs one picture from the small 'lores' stream and returns its brightness (grey) part.

    The lores stream comes in YUV420 format: the first LORES_SIZE[1] rows are
    the brightness, which is all that sharpness needs.
    """
    width, height = LORES_SIZE
    return cam.capture_array("lores")[:height, :width]


def autofocus(cam, bus, grab=None, report=None):
    """Finds the sharpest focus setting with the camera already running, and moves the lens there.

    cam    : a started Picamera2 camera with a lores stream
    bus    : the camera's I2C bus number
    grab   : function returning a grey picture (default: from the lores stream)
    report : optional function called with (setting, sharpness) after each try

    Returns a dictionary with the best setting, its sharpness, and at_edge,
    which is True when the best setting is at the end of the motor's range.
    """
    grab = grab or (lambda: lores_gray(cam))
    scores = {}

    def measure(value):
        if value not in scores:
            set_focus(value, bus)
            time.sleep(SETTLE_S)
            grab()                                   # throw away a picture taken while the lens moved
            scores[value] = sharpness_gray(grab())
            if report:
                report(value, scores[value])
        return scores[value]

    for value in list(range(FOCUS_MIN, FOCUS_MAX + 1, COARSE_STEP)) + [FOCUS_MAX]:
        measure(value)
    coarse_best = max(scores, key=scores.get)
    for value in range(max(FOCUS_MIN, coarse_best - COARSE_STEP), min(FOCUS_MAX, coarse_best + COARSE_STEP) + 1, FINE_STEP):
        measure(value)

    best = max(scores, key=scores.get)
    set_focus(best, bus)
    time.sleep(SETTLE_S)
    return {
        "value": best,
        "sharpness": scores[best],
        "at_edge": best <= FOCUS_MIN + EDGE_MARGIN or best >= FOCUS_MAX - EDGE_MARGIN,
    }


def describe(result):
    """A one-line description of an autofocus result for people to read."""
    text = f"focus setting {result['value']}, sharpness {result['sharpness']:.0f}"
    if result["at_edge"]:
        text += (" (at the END of the motor's range: the lens needs turning by hand; "
                 "run 'python3 focus.py tune')")
    return text


# ---------------------------------------------------------------------------
# Terminal commands
# ---------------------------------------------------------------------------
def _open_camera():
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(
        main={"size": cam.sensor_resolution, "format": "RGB888"},
        lores={"size": LORES_SIZE},
    ))
    cam.start()
    time.sleep(2)                 # let exposure and white balance settle
    return cam


def sweep():
    """Runs autofocus once, saves the result and takes focus_test.jpg."""
    bus = find_camera_bus()
    cam = _open_camera()
    try:
        print("Autofocus running. Keep your hands out of view...")
        result = autofocus(cam, bus, report=lambda v, s: print(f"  focus {v:4d}: sharpness {s:9.1f}"))
        cam.capture_array("main")
        cv2.imwrite("focus_test.jpg", cam.capture_array("main"))
    finally:
        cam.stop()
        cam.close()
    save_focus(result["value"])
    print(f"\nResult: {describe(result)}\nSaved, and photo saved as focus_test.jpg.")


def tune():
    """Helps set the lens by hand without your hand spoiling the measurement.

    Each round: turn the lens a little, take your hand away, press Enter. The
    script then runs autofocus with the motor and reports the best sharpness it
    can reach at this lens position, and whether the motor's range now covers
    the sharp point. Type q and Enter to finish.
    """
    bus = find_camera_bus()
    cam = _open_camera()
    best_ever = 0.0
    print("Put a printed PCB sheet under the camera at the distance you will use.\n"
          "Each round: turn the lens a little (about 1/8 turn), take your hand away, press Enter.\n"
          "Type q and Enter to finish.\n")
    try:
        round_no = 0
        while True:
            answer = input(f"Round {round_no}: press Enter to measure (q to finish): ").strip().lower()
            if answer == "q":
                break
            result = autofocus(cam, bus)
            better = result["sharpness"] > best_ever
            best_ever = max(best_ever, result["sharpness"])
            print(f"   {describe(result)}")
            print("   BETTER than before: keep turning the same way." if better and round_no > 0
                  else "   worse than the best so far: turn back the other way." if round_no > 0
                  else "   now turn the lens a little in either direction.")
            if not result["at_edge"] and better:
                save_focus(result["value"])
                print("   The motor can reach this sharp point. If the picture is good, type q.")
            round_no += 1
    finally:
        cam.stop()
        cam.close()
    print(f"Finished. Best sharpness reached: {best_ever:.0f}. Now run: python3 focus.py sweep")


def photo():
    """Takes focus_test.jpg using the saved focus setting."""
    value = saved_focus()
    if value is None:
        sys.exit("No saved focus setting. Run: python3 focus.py sweep")
    bus = find_camera_bus()
    cam = _open_camera()
    try:
        set_focus(value, bus)
        time.sleep(SETTLE_S)
        cam.capture_array("main")
        frame = cam.capture_array("main")
    finally:
        cam.stop()
        cam.close()
    cv2.imwrite("focus_test.jpg", frame)
    print(f"focus_test.jpg taken at focus {value} (sharpness {sharpness(frame):.1f}).")


def main():
    args = sys.argv[1:]
    if args == ["sweep"]:
        sweep()
    elif args == ["tune"]:
        tune()
    elif args == ["photo"]:
        photo()
    else:
        sys.exit("Usage: python3 focus.py sweep | tune | photo")


if __name__ == "__main__":
    main()
