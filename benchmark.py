"""Measures how fast the station runs on this computer, for the project report.

What it measures, using the same functions as the dashboard:
  - loading time of the PCB check and the defect detector
  - PCB check time and defect search time for each test picture (with tile count)
  - Llama advisory on this computer: time to first token, tokens per second,
    total time and peak memory (as reported by llama_runner.py)
  - memory used and CPU temperature before and after

How to run it on the Pi (stop the dashboard first, so the camera and the CPU are free):

    cd ~/BharatAIHackathon
    source pcb_env/bin/activate
    python3 benchmark.py ~/sample_images            # a folder of test pictures
    python3 benchmark.py ~/sample_images --camera   # also time one real camera photo
    python3 benchmark.py ~/sample_images --no-llama # skip the Llama part

The results are printed and also saved to benchmark_results.txt.
"""
import argparse
import glob
import os
import platform
import statistics
import time

import cv2
import psutil

import llm_engine
import vision

IMAGE_TYPES = (".jpg", ".jpeg", ".png", ".bmp")
RESULTS_FILE = "benchmark_results.txt"


def cpu_temperature():
    """CPU temperature in degrees C, or None on computers that do not report it."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000
    except (OSError, ValueError):
        return None


def device_name():
    """'Raspberry Pi 5 Model B Rev 1.0' on a Pi, otherwise the operating system and CPU."""
    try:
        with open("/proc/device-tree/model") as f:
            return f.read().strip("\x00\n ")
    except OSError:
        return f"{platform.system()} {platform.machine()}"


def find_images(paths):
    """Turns the folders and files given on the command line into a list of picture files."""
    found = []
    for p in paths:
        if os.path.isdir(p):
            found += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(IMAGE_TYPES))
        elif p.lower().endswith(IMAGE_TYPES):
            found.append(p)
    return found


def summary(values, unit, digits=0):
    """'mean 512 ms (min 498, max 530, n=12)' for a list of numbers."""
    if not values:
        return "not measured"
    f = f".{digits}f"
    return (f"mean {statistics.mean(values):{f}} {unit} "
            f"(min {min(values):{f}}, max {max(values):{f}}, n={len(values)})")


def main():
    parser = argparse.ArgumentParser(description="Benchmark the PCB inspection station.")
    parser.add_argument("images", nargs="*", help="test pictures, or folders of them")
    parser.add_argument("--camera", action="store_true", help="also capture and time one camera photo")
    parser.add_argument("--no-llama", action="store_true", help="skip the Llama measurement")
    parser.add_argument("--llama-runs", type=int, default=3, help="how many Llama answers to time")
    args = parser.parse_args()

    lines = []

    def report(text=""):
        print(text, flush=True)
        lines.append(text)

    report(f"Device       : {device_name()}")
    report(f"Date         : {time.strftime('%Y-%m-%d %H:%M')}")
    report(f"RAM in use   : {psutil.virtual_memory().used / 2**20:.0f} MB of {psutil.virtual_memory().total / 2**20:.0f} MB")
    temp_start = cpu_temperature()
    report(f"CPU temp     : {temp_start} C (start)")
    report()

    # --- loading the two vision models ---
    t0 = time.perf_counter()
    model, model_path = vision.load_model()
    gate = vision.load_gate()
    load_ms = (time.perf_counter() - t0) * 1000
    report(f"Defect model : {model_path}")
    report(f"PCB check    : {vision.GATE_MODEL if gate else 'not found (skipped)'}")
    report(f"Load time    : {load_ms:.0f} ms for both models")

    # --- the test pictures ---
    pictures = [(path, cv2.imread(path)) for path in find_images(args.images)]
    pictures = [(path, img) for path, img in pictures if img is not None]
    if args.camera:
        import camera                      # only needed here, and it opens the camera
        frame = camera.capture_frame()
        if frame is not None:
            pictures.append((f"camera photo ({camera.camera_name()})", frame))
    if not pictures:
        report("\nNo test pictures given. Example: python3 benchmark.py ~/sample_images")
        return

    is_pcb_model = model_path == vision.PCB_MODEL
    vision.detect(model, pictures[0][1], use_tiles=is_pcb_model)      # warm-up run, not counted
    if gate:
        vision.pcb_probability(gate, pictures[0][1])

    report()
    report(f"{'Picture':34s} {'Size':>11s} {'Tiles':>5s} {'p(PCB)':>6s} {'Check ms':>8s} {'Search ms':>9s} {'Defects':>7s}")
    gate_times, search_times, per_tile = [], [], []
    first_with_defects = None
    for path, img in pictures:
        height, width = img.shape[:2]
        prob, gate_ms = None, None
        if gate:
            t0 = time.perf_counter()
            prob = vision.pcb_probability(gate, img)
            gate_ms = (time.perf_counter() - t0) * 1000
            gate_times.append(gate_ms)
        found = vision.detect(model, img, use_tiles=is_pcb_model)
        search_times.append(found["vision_ms"])
        per_tile.append(found["vision_ms"] / found["tiles"])
        if found["detections"] and first_with_defects is None:
            first_with_defects = llm_engine.add_severity_and_location(found["detections"], width, height)
        report(f"{os.path.basename(path)[:34]:34s} {f'{width}x{height}':>11s} {found['tiles']:5d} "
               f"{'-' if prob is None else f'{prob:.2f}':>6s} {'-' if gate_ms is None else f'{gate_ms:.0f}':>8s} "
               f"{found['vision_ms']:9.0f} {len(found['detections']):7d}")

    report()
    report(f"PCB check time    : {summary(gate_times, 'ms')}")
    report(f"Defect search time: {summary(search_times, 'ms')}")
    report(f"Time per tile     : {summary(per_tile, 'ms')}")
    report(f"Vision process RAM: {psutil.Process().memory_info().rss / 2**20:.0f} MB")
    report(f"CPU temp          : {cpu_temperature()} C (after vision)")

    # --- Llama advisory ---
    if args.no_llama:
        report("\nLlama: skipped (--no-llama)")
    elif not llm_engine.llama_available():
        report("\nLlama: not installed on this computer, skipped")
    elif first_with_defects is None:
        report("\nLlama: no picture had defects, so there was nothing to advise on")
    else:
        report()
        report(f"Llama findings    : {llm_engine.summary_line(first_with_defects)}")
        ttft, speed, total, memory, sources = [], [], [], [], []
        for _ in range(args.llama_runs):
            advice = llm_engine.advise(first_with_defects)
            stats = advice.get("stats", {})
            sources.append(advice["source"])
            if stats.get("time_to_first_token_ms") is not None:
                ttft.append(stats["time_to_first_token_ms"])
            if stats.get("tokens_per_second"):
                speed.append(stats["tokens_per_second"])
            if stats.get("total_ms_including_model_load"):
                total.append(stats["total_ms_including_model_load"])
            if stats.get("peak_memory_mb"):
                memory.append(stats["peak_memory_mb"])
        report(f"Answer used       : {', '.join(sources)}  (llama = Llama's answer passed the checks)")
        report(f"Last instruction  : {advice['text']}")
        report(f"Time to 1st token : {summary(ttft, 'ms')}")
        report(f"Speed             : {summary(speed, 'tokens/s', 1)}")
        report(f"Total incl. load  : {summary(total, 'ms')}")
        report(f"Llama peak memory : {summary(memory, 'MB')}")
        report(f"CPU temp          : {cpu_temperature()} C (after Llama)")

    with open(RESULTS_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
