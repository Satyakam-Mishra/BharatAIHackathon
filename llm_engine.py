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
import re
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
    "You write short notes for operators on a bare PCB inspection line. "
    "The defects were found by a vision model and their severities were set by fixed rules, "
    "so never change them. Use only the facts and reference notes in the message. "
    "Answer with exactly three lines that start with Summary:, Cause: and Action:, one sentence each."
)

# One worked example shown to Llama before the real question. A model this small
# follows a pattern it has just seen far more reliably than a written instruction.
EXAMPLE_FACTS = [
    {"defect": "missing_hole", "severity": "High", "location": "top-left", "size_pct": 2.1, "confidence": 0.88},
    {"defect": "mouse_bite", "severity": "Medium", "location": "centre", "size_pct": 1.6, "confidence": 0.79},
]
EXAMPLE_ANSWER = (
    "Summary: The board fails with 2 defects, a High severity missing_hole at top-left and a "
    "Medium severity mouse_bite at centre.\n"
    "Cause: The missing hole points to a skipped drill hit or a wrong drill program, and the "
    "mouse bite to over-etching at the edge of a track.\n"
    "Action: Compare the board with the drill file and re-drill or reject it, then measure the "
    "remaining track width at the mouse bite."
)
ANSWER_PREFIX = "Summary:"       # the answer is forced to begin with this word
NOTE_LABELS = ("Summary:", "Cause:", "Action:")


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
    """Writes the findings and the matching reference notes as plain sentences for Llama.

    Defects of the same type and severity are grouped into one sentence, so a
    board with ten spurs still gives a short prompt.
    """
    groups = {}
    for f in findings:
        groups.setdefault((f["defect"], f["severity"]), []).append(f)
    ordered = sorted(groups.items(), key=lambda item: LEVELS.index(item[0][1]), reverse=True)

    lines = [f"Facts: {len(findings)} defect(s) found, highest severity {highest_severity(findings)}."]
    for (defect, severity), members in ordered:
        places = []
        for m in members:
            if m["location"] not in places:
                places.append(m["location"])
        confidences = [m["confidence"] for m in members]
        confidence = f"{min(confidences)}" if len(members) == 1 else f"{min(confidences)} to {max(confidences)}"
        lines.append(
            f"{len(members)} {defect} (severity {severity}) at {', '.join(places)}; "
            f"largest is {max(m['size_pct'] for m in members)}% of the board; confidence {confidence}."
        )

    lines.append("Reference notes:")
    for (defect, _), _ in ordered:
        if defect in DEFECT_NOTES:
            note = DEFECT_NOTES[defect]
            line = f"- {defect}: cause: {note['cause']}; action: {note['action']}."
            if line not in lines:
                lines.append(line)
    return "\n".join(lines)


def parse_note(answer):
    """Pulls the Summary, Cause and Action sentences out of Llama's answer.

    Small models sometimes add a greeting, repeat the input or keep talking after
    the third line. We keep the first sentence found for each label and drop the rest.
    Raises RuntimeError if any of the three is missing.
    """
    found = {}
    for chunk in re.split(r"(?=(?:Summary|Cause|Action):)", answer):
        chunk = chunk.strip()
        for label in NOTE_LABELS:
            if chunk.startswith(label) and label not in found:
                first_line = chunk.splitlines()[0].strip()
                if len(first_line) > len(label) + 10:       # ignore an empty or cut-off line
                    found[label] = first_line
    missing = [label for label in NOTE_LABELS if label not in found]
    if missing:
        raise RuntimeError(f"Llama's answer had no usable {' / '.join(missing)} line")
    return "\n".join(found[label] for label in NOTE_LABELS)


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


def ask_llama(user_text, max_new_tokens=110):
    """Runs llama_runner.py in the ExecuTorch environment and returns its result dictionary.

    The runner is a separate program, so we send the request as JSON on its standard
    input and look for the line it prints that starts with RESULT_JSON:.
    Raises RuntimeError if anything goes wrong.
    """
    request = json.dumps({
        "system": SYSTEM_PROMPT,
        "examples": [[build_user_prompt(EXAMPLE_FACTS), EXAMPLE_ANSWER]],
        "user": user_text,
        "assistant_prefix": ANSWER_PREFIX,
        "max_new_tokens": max_new_tokens,
    })
    proc = subprocess.run([ET_PYTHON, RUNNER], input=request, capture_output=True, text=True, timeout=LLAMA_TIMEOUT_S)
    start = proc.stdout.rfind(RESULT_MARKER)
    if start < 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise RuntimeError(f"Llama runner gave no result (exit code {proc.returncode}): {tail[0][:200]}")
    payload = proc.stdout[start + len(RESULT_MARKER):].splitlines()[0]
    result = json.loads(payload)
    # The runner returns only the words Llama added after the forced first word.
    result["text"] = ANSWER_PREFIX + " " + result["text"].strip()
    return result


def advise(findings):
    """Turns the findings into an operator note.

    Returns a dictionary:
      text   : the note
      source : 'llama' (generated on-device by Llama via ExecuTorch), 'rules'
               (written by the fixed rules) or 'none' (clean board)
      stats  : speed figures from Llama, when it was used
      raw    : Llama's full answer before tidying, when it was used
      error  : why Llama was not used, when that happened
    """
    if not findings:
        return {"text": "No surface defects detected. Pass the board to the next stage.", "source": "none", "stats": {}}

    if not llama_available():
        return {"text": rule_based_note(findings), "source": "rules", "stats": {},
                "error": "Llama is not installed on this computer"}

    try:
        result = ask_llama(build_user_prompt(findings))
        return {"text": parse_note(result["text"]), "source": "llama", "stats": result.get("stats", {}),
                "raw": result["text"]}
    except Exception as exc:                      # timeout, crash, or an answer in the wrong shape
        return {"text": rule_based_note(findings), "source": "rules", "stats": {}, "error": str(exc)}


if __name__ == "__main__":
    # Quick test without the dashboard:  python3 llm_engine.py
    # Three different boards, to see how Llama copes with each.
    SIZE = (3034, 1586)
    cases = {
        "A: three spurs": [
            {"defect": "spur", "confidence": 0.64, "box": [1480, 400, 1538, 450]},
            {"defect": "spur", "confidence": 0.72, "box": [1800, 380, 1845, 420]},
            {"defect": "spur", "confidence": 0.75, "box": [1830, 560, 1870, 600]},
        ],
        "B: one short and two spurs": [
            {"defect": "short", "confidence": 0.91, "box": [2400, 300, 2470, 360]},
            {"defect": "spur", "confidence": 0.72, "box": [1500, 800, 1560, 850]},
            {"defect": "spur", "confidence": 0.64, "box": [400, 1200, 450, 1260]},
        ],
        "C: open circuit, large spurious copper, mouse bite": [
            {"defect": "open_circuit", "confidence": 0.83, "box": [300, 700, 360, 740]},
            {"defect": "spurious_copper", "confidence": 0.77, "box": [2000, 1100, 2120, 1180]},
            {"defect": "mouse_bite", "confidence": 0.69, "box": [1500, 200, 1550, 250]},
        ],
    }
    for name, detections in cases.items():
        case_findings = add_severity_and_location(detections, *SIZE)
        print(f"\n================ Case {name} ================")
        print(build_user_prompt(case_findings))
        answer = advise(case_findings)
        print(f"\n--- Note shown to the operator (source: {answer['source']}) ---")
        print(answer["text"])
        if answer.get("raw"):
            print("\n--- Llama's full answer before tidying ---")
            print(answer["raw"])
        if answer.get("error"):
            print("\nWhy Llama was not used:", answer["error"])
        if answer["stats"]:
            print("\n--- Speed ---")
            for key, value in answer["stats"].items():
                print(f"{key:32s} {value}")
