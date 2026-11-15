"""Advisory pipeline, dashboard side.

What happens here, in order:
  1. Fixed RULES give every detected defect a severity and a location name.
     (Rules decide; the language model does not.)
  2. The findings are written into a short, structured prompt.
  3. Llama 3.2 1B, running on ExecuTorch in its own Python environment, turns
     that prompt into a three-line note for the operator (see llama_runner.py).
  4. If Llama is not installed on this computer (for example a Windows laptop),
     a plain rule-based note is produced instead, and it is labelled as such.
"""
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Where the ExecuTorch side lives on the Raspberry Pi
# ---------------------------------------------------------------------------
ET_PYTHON = os.path.expanduser("~/et_env/bin/python")
RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_runner.py")
MODEL_FILE = os.path.expanduser("~/llama/llama3_2_1b_main.pte")
RESULT_MARKER = "RESULT_JSON:"
LLAMA_TIMEOUT_S = 120

# ---------------------------------------------------------------------------
# Severity rules (document these in the report)
#   Step 1: a base level from the defect TYPE, by its effect on the circuit.
#           High   = the circuit is electrically or mechanically wrong
#           Medium = it works but with reduced margin (width or clearance)
#           Low    = cosmetic or isolated, usually no functional effect
#   Step 2: SIZE. A defect whose longest side is more than LARGE_DEFECT_FRACTION
#           of the picture's longest side is raised by one level.
#   Confidence is NOT used: it says how sure the model is, not how bad the defect is.
# ---------------------------------------------------------------------------
LEVELS = ["Low", "Medium", "High"]
BASE_SEVERITY = {
    "open_circuit": "High",
    "short": "High",
    "missing_hole": "High",
    "mouse_bite": "Medium",
    "spur": "Medium",
    "spurious_copper": "Low",
}
LARGE_DEFECT_FRACTION = 0.03

# Short reference notes given to Llama so its advice is grounded in facts
# instead of guesses. One likely cause and one action per defect type.
DEFECT_NOTES = {
    "missing_hole": {
        "cause": "a drill hit was skipped or the wrong drill program was used",
        "action": "compare the board with the drill file, re-drill if the pad allows it, otherwise reject the board",
    },
    "mouse_bite": {
        "cause": "over-etching or a flaw in the resist removed copper from the edge of a track",
        "action": "measure the remaining track width and repair or reject if it is below the design minimum",
    },
    "open_circuit": {
        "cause": "a scratch in the resist or over-etching broke the track",
        "action": "confirm with a continuity test, then repair the track if repairs are allowed, otherwise reject the board",
    },
    "short": {
        "cause": "under-etching or resist residue left copper bridging two tracks",
        "action": "cut the copper bridge under magnification and confirm isolation with a multimeter",
    },
    "spur": {
        "cause": "under-etching or a resist flaw left a copper spike sticking out of a track",
        "action": "check the clearance to the neighbouring track and trim the spike if it is below the design rule",
    },
    "spurious_copper": {
        "cause": "resist residue protected a patch of copper that should have been etched away",
        "action": "remove the copper if it is close to a track or pad, otherwise record it and accept the board",
    },
}

SYSTEM_PROMPT = (
    "You are a quality-control assistant on a bare PCB inspection line. "
    "A vision model has already found the defects and fixed rules have set each severity. "
    "Do not change the defect types, counts or severities, and use only the findings and "
    "reference notes you are given. Reply with exactly three lines and nothing else:\n"
    "Summary: one sentence saying what was found and the highest severity.\n"
    "Cause: one sentence giving the most likely process cause.\n"
    "Action: one sentence telling the operator what to do next."
)


def _location_name(cx, cy, width, height):
    """Names the part of the picture a point is in, on a 3 x 3 grid."""
    col = ["left", "centre", "right"][min(2, int(3 * cx / width))]
    row = ["top", "middle", "bottom"][min(2, int(3 * cy / height))]
    if row == "middle" and col == "centre":
        return "centre"
    return f"{row}-{col}"


def add_severity_and_location(detections, width, height):
    """Returns the detections with three rule-based fields added to each:
    severity (Low/Medium/High), location (for example 'top-right') and
    size_pct (longest side of the box as a percentage of the picture's longest side).
    """
    findings = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        longest_side = max(x2 - x1, y2 - y1)
        fraction = longest_side / max(width, height)

        level = LEVELS.index(BASE_SEVERITY.get(det["defect"], "Medium"))
        if fraction > LARGE_DEFECT_FRACTION:
            level = min(level + 1, len(LEVELS) - 1)

        finding = dict(det)
        finding["severity"] = LEVELS[level]
        finding["location"] = _location_name((x1 + x2) / 2, (y1 + y2) / 2, width, height)
        finding["size_pct"] = round(100 * fraction, 1)
        findings.append(finding)
    return findings


def highest_severity(findings):
    """The worst severity among the findings, or None for a clean board."""
    if not findings:
        return None
    return max((f["severity"] for f in findings), key=LEVELS.index)


def build_user_prompt(findings):
    """Writes the findings and the matching reference notes as plain text for Llama."""
    worst_first = sorted(findings, key=lambda f: LEVELS.index(f["severity"]), reverse=True)
    lines = [f"Board verdict: FAIL. Defects found: {len(findings)}. Highest severity: {highest_severity(findings)}.",
             "Findings:"]
    for i, f in enumerate(worst_first[:8], start=1):      # at most 8 lines keeps the prompt short
        lines.append(
            f"{i}. {f['defect']} | severity {f['severity']} | location {f['location']} | "
            f"size {f['size_pct']}% of board | confidence {f['confidence']}"
        )
    if len(worst_first) > 8:
        lines.append(f"(and {len(worst_first) - 8} more of lower or equal severity)")

    lines.append("Reference notes:")
    seen = []
    for f in worst_first:
        if f["defect"] in DEFECT_NOTES and f["defect"] not in seen:
            seen.append(f["defect"])
            note = DEFECT_NOTES[f["defect"]]
            lines.append(f"- {f['defect']}: likely cause: {note['cause']}. Action: {note['action']}.")
    return "\n".join(lines)


def rule_based_note(findings):
    """The same three-line note written without a language model."""
    worst = max(findings, key=lambda f: LEVELS.index(f["severity"]))
    note = DEFECT_NOTES.get(worst["defect"], {"cause": "unknown", "action": "send the board for manual inspection"})
    types = ", ".join(sorted({f["defect"] for f in findings}))
    return (
        f"Summary: {len(findings)} defect(s) found ({types}); highest severity {worst['severity']}.\n"
        f"Cause: {worst['defect']} is most likely because {note['cause']}.\n"
        f"Action: {note['action'][0].upper()}{note['action'][1:]}."
    )


def llama_available():
    """True when the ExecuTorch environment, the runner script and the model file all exist."""
    return all(os.path.exists(p) for p in (ET_PYTHON, RUNNER, MODEL_FILE))


def ask_llama(system_text, user_text, max_new_tokens=120):
    """Runs llama_runner.py in the ExecuTorch environment and returns its result dictionary.

    The runner is a separate program, so we send the request as JSON on its standard
    input and look for the line it prints that starts with RESULT_JSON:.
    Raises RuntimeError if anything goes wrong.
    """
    request = json.dumps({"system": system_text, "user": user_text, "max_new_tokens": max_new_tokens})
    proc = subprocess.run([ET_PYTHON, RUNNER], input=request, capture_output=True, text=True, timeout=LLAMA_TIMEOUT_S)
    start = proc.stdout.rfind(RESULT_MARKER)
    if start < 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"Llama runner gave no result (exit code {proc.returncode}): {tail[0][:200]}")
    payload = proc.stdout[start + len(RESULT_MARKER):].splitlines()[0]
    return json.loads(payload)


def advise(findings):
    """Turns the findings into an operator note.

    Returns a dictionary:
      text   : the note
      source : 'llama' (generated on-device by Llama via ExecuTorch), 'rules'
               (written by the fixed rules) or 'none' (clean board)
      stats  : speed figures from Llama, when it was used
      error  : why Llama was not used, when that happened
    """
    if not findings:
        return {"text": "No surface defects detected. Pass the board to the next stage.", "source": "none", "stats": {}}

    if not llama_available():
        return {"text": rule_based_note(findings), "source": "rules", "stats": {},
                "error": "Llama is not installed on this computer"}

    try:
        result = ask_llama(SYSTEM_PROMPT, build_user_prompt(findings))
        text = "\n".join(line.strip() for line in result["text"].splitlines() if line.strip())
        if not text:
            raise RuntimeError("Llama returned an empty answer")
        return {"text": text, "source": "llama", "stats": result.get("stats", {})}
    except Exception as exc:                      # timeout, crash, unreadable output
        return {"text": rule_based_note(findings), "source": "rules", "stats": {}, "error": str(exc)}


if __name__ == "__main__":
    # Quick test without the dashboard:  python3 llm_engine.py
    sample = [
        {"defect": "short", "confidence": 0.91, "box": [2400, 300, 2470, 360]},
        {"defect": "spur", "confidence": 0.72, "box": [1500, 800, 1560, 850]},
        {"defect": "spur", "confidence": 0.64, "box": [400, 1200, 450, 1260]},
    ]
    sample_findings = add_severity_and_location(sample, 3034, 1586)
    print("--- Prompt sent to Llama ---")
    print(build_user_prompt(sample_findings))
    answer = advise(sample_findings)
    print(f"\n--- Advisory (source: {answer['source']}) ---")
    print(answer["text"])
    if answer.get("error"):
        print("\nNote:", answer["error"])
    if answer["stats"]:
        print("\n--- Speed ---")
        for key, value in answer["stats"].items():
            print(f"{key:32s} {value}")
    sys.exit(0)
