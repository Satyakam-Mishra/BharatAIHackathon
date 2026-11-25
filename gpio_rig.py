import time

# Attempt to import Raspberry Pi 5 GPIO library; fall back to simulation on Windows/Mac
try:
    from gpiozero import LED
    # GPIO 17 (Pin 11) = Green Pass LED | GPIO 27 (Pin 13) = Red Reject LED/Buzzer
    green_led = LED(17)
    red_led = LED(27)
    IS_RPI_GPIO = True
except ImportError:
    IS_RPI_GPIO = False

def trigger_inspection_signal(has_defects):
    """
    Drives physical pass/fail status LEDs on the Raspberry Pi 5 GPIO header.
    - Green LED (GPIO 17): Lit when PCB is nominal (0 defects).
    - Red LED (GPIO 27): Lit when PCB has surface defects requiring rework.
    """
    if IS_RPI_GPIO:
        if has_defects:
            green_led.off()
            red_led.on()
        else:
            red_led.off()
            green_led.on()
    else:
        status = "RED LED ON (Defect Reject)" if has_defects else "GREEN LED ON (Pass)"
        print(f"[GPIO Rig Simulation] -> {status}")