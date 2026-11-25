# Task: train the "no PCB in view" check

**For:** the teammate taking this over. **Written:** 24 November 2026.

## What this is and why

Sir asked that the station give a third answer, **"no PCB in view"**, instead of PASS or FAIL when there is no board under the camera. Right now the defect detector looks at whatever the camera sees and can report defects on a desk or a hand.

The code for this is already written and on GitHub. What is missing is the trained model file, `pcb_gate.onnx`. It is a very small image classifier (YOLOv8n-cls) with two classes, `pcb` and `not_pcb`, that looks at the whole picture first:

- **PCB:** the normal inspection runs and gives PASS or FAIL.
- **Not a PCB:** the dashboard shows "NO PCB IN VIEW: neither PASS nor FAIL", skips the defect search and Llama, and logs the row as "No PCB".

Until `pcb_gate.onnx` is on the Pi, the dashboard shows a notice that the check is skipped and treats every picture as a PCB.

The relevant files in the repo:

| File | What it does |
|---|---|
| `training/train_pcb_gate_colab.ipynb` | Colab notebook that trains the check and gives you `pcb_gate.onnx` |
| `collect_photos.py` | Takes photos with the station's camera, for the optional later improvement |
| `vision.py` | `load_gate()` and `pcb_probability()`: runs the check on the Pi (threshold 0.60) |
| `dashboard.py` | Shows the third state and logs it |

**Training data (current plan):** downloaded and generated pictures only. The notebook downloads 693 PCB photos and 1,000 everyday photos, and generates "board lying on a table" pictures and plain scenes (blank paper, plain surfaces, shadows). No photos need to be collected. Photos from the station camera are an optional improvement for later (see the end).

Total time: about 45 minutes, most of it waiting.

## Before you start

- **Raspberry Pi:** hostname `pcb-pi`, user `pi`. Ask Aryan for the password.
- **Network:** your laptop and the Pi must be on the same Wi-Fi or hotspot. The Pi joins the hotspot it was set up with; ask Aryan which one.
- **Camera:** the Arducam IMX477 is mounted on a stand pointing down. **Do not change its height.** If it does move, press "Refocus camera" in the dashboard with your hands out of view.
- **Accounts:** a Google account for Colab. The downloads in the notebook need no login.

## Step 1: Connect to the Pi and update the code

On your laptop (PowerShell on Windows, Terminal on Mac):

```bash
ssh pi@pcb-pi.local
```

Then on the Pi:

```bash
cd ~/BharatAIHackathon
source pcb_env/bin/activate
git pull
```

Only one program can use the camera at a time. If anything else is running, stop it first:

```bash
pkill -f streamlit; pkill -f focus.py
```

## Step 2: Train on Colab (about 20 to 30 minutes)

1. Use the notebook file `train_pcb_gate_colab.ipynb` that was sent with this guide (it is also in the repo under `training/`).
2. Go to colab.research.google.com, choose **File → Upload notebook** and pick it.
3. Choose **Runtime → Change runtime type → T4 GPU → Save**.
4. Run the cells from top to bottom with **Shift + Enter**. Each cell has a short explanation above it.
   - In the notebook's Step 3 cell, leave `USE_OWN_PHOTOS = False`.
   - Notebook Step 6 shows a picture: the top row should show PCBs, the bottom row not.
   - Notebook Step 8 prints the score on the held-back test pictures. **Copy those lines.** They are a sanity check only:
     every test picture is downloaded or generated, so the real test is Step 4 of this guide, at the station.
5. Notebook Step 9 downloads `pcb_gate.zip`. Unzip it.

## Step 3: Put the model on the Pi

From the unzipped folder on your laptop:

```bash
scp pcb_gate.onnx pi@pcb-pi.local:~/BharatAIHackathon/
```

Do not commit `.onnx` files to Git; the repo ignores them on purpose.

## Step 4: Test all three outcomes at the station

On the Pi:

```bash
cd ~/BharatAIHackathon
source pcb_env/bin/activate
streamlit run dashboard.py --server.address 0.0.0.0 --server.headless true
```

Open `http://pcb-pi.local:8501` in your laptop's browser. The first load takes about 10 seconds longer while the camera focuses. The notice about the PCB check being skipped should be gone.

Use **Capture and inspect** with:

| Under the camera | Expected |
|---|---|
| Nothing (empty stand) | "NO PCB IN VIEW", grey caption "PCB check: 0.xx \| Defect search skipped" |
| A hand, phone or text page | "NO PCB IN VIEW" |
| A printed demo sheet | FAIL with defects (vision takes about 7 to 8 s) |
| A board | PASS or FAIL (see the note below) |

Try each case a few times, with things in different places. Take a screenshot of each.

**Note:** on our own fabricated board the defect detector currently raises false alarms (it was trained on a different dataset). That is a known limitation, separate from this task. This task only has to tell "PCB" from "not a PCB" correctly.

## Step 5: Send back to the team

- The notebook's Step 8 lines and `gate_metrics.txt` from the zip (for the report)
- The screenshots from Step 4
- Any case where it got "PCB / not a PCB" wrong, with a photo

**If it gets things wrong at the station** (likely cases: the empty stand or blank paper called a PCB, or a board called "no PCB"), use the improvement below.

## Later improvement: add photos from the station camera

This makes the check far more reliable, because it then learns from exactly what the station sees.

1. On the Pi, with the dashboard stopped:
   ```bash
   python3 collect_photos.py pcb 20
   python3 collect_photos.py not_pcb 20
   ```
   A photo is taken every 4 seconds; change the scene between shots. For `pcb`: every board you have, moved and rotated (not the 12 demo sheets). For `not_pcb`: the empty stand, blank paper, a text page, a hand, a phone, a pen. Each takes about 1.5 minutes.
2. Copy them to your laptop: `scp -r pi@pcb-pi.local:~/gate_photos .`
3. In Google Drive make `My Drive/pcb_gate_photos/` and upload the `pcb` and `not_pcb` folders into it.
4. In the notebook, set `USE_OWN_PHOTOS = True` in its Step 3 cell and run it again. Its Step 8 then also scores your own held-back photos.

## Shutting down

1. Stop the dashboard with `Ctrl + C`.
2. Run `sudo shutdown now`, wait about 15 seconds for the Pi's LED to stay steady red, then unplug it.
3. In Colab: **Runtime → Disconnect and delete runtime**.

## If something goes wrong

| Problem | Fix |
|---|---|
| `ssh` can't find `pcb-pi.local` | Check the laptop is on the same hotspot as the Pi; or use the Pi's IP address shown in the hotspot's device list |
| "Pipeline handler in use by another process" | Another program has the camera: `pkill -f streamlit; pkill -f focus.py; pkill -f collect_photos.py` |
| Pictures are blurry | Press "Refocus camera" in the dashboard with your hands out of view |
| Notebook Step 3 says fewer than 10 photos | Only when `USE_OWN_PHOTOS = True`: check the Drive folder names and that both subfolders have photos |
| pip says "No space left on device" | Run `export TMPDIR=~/piptmp` first |
