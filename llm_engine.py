"""Advisory pipeline, dashboard side.

What happens here, in order:
  1. Fixed RULES give every detected defect a severity and a location name.
     (Rules decide; the language model does not.)
  2. The code writes the factual summary line itself, so the counts, types,
     severities and locations shown to the operator are always exact.
  3. Llama 3.2 1B, running on ExecuTorch in its own Python environment, writes
     the instruction for the operator: what to do and where, most severe defect
     first (see llama_runner.py).
  4. Llama's instruction is checked. If it is cut off, leaves out a defect or a
     location that was found, or names a defect that was not found, or if Llama is not installed
     on this computer, a plain rule-based instruction is shown instead and it is
     labelled as such.
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

# The standard action for each defect type. Llama is given these so that its
# instruction is grounded in facts instead of guesses.
DEFECT_ACTIONS = {
    "missing_hole": "compare the board with the drill file, re-drill if the pad allows it, otherwise reject the board",
    "mouse_bite": "measure the remaining track width and repair or reject if it is below the design minimum",
    "open_circuit": "confirm with a continuity test, then repair the track if repairs are allowed, otherwise reject the board",
    "short": "cut the copper bridge under magnification and confirm isolation with a multimeter",
    "spur": "check the clearance to the neighbouring track and trim the spike if it is below the design rule",
    "spurious_copper": "remove the copper if it is close to a track or pad, otherwise record it and accept the board",
}
UNKNOWN_ACTION = "send the board for manual inspection"

# How each defect type may be written in a sentence: with a space, hyphen or
# underscore, and in the singular or plural ("missing_hole", "missing holes").
# \b marks a word boundary, so "spur" does not match inside "spurious".
DEFECT_PATTERNS = {
    name: re.compile(r"\b" + name.replace("_", "[ _-]") + r"s?\b", re.IGNORECASE) for name in DEFECT_ACTIONS
}

SYSTEM_PROMPT = (
    "You write the instruction an operator sees when a bare PCB fails inspection. "
    "You are given the defects, already ordered from most to least severe, each with its standard action. "
    "Write one instruction of one to three sentences that tells the operator what to do and where on the board, "
    "starting with the most severe defect. Use only the information given. "
    "Do not mention causes, sizes or confidence."
)

# Two worked examples shown to Llama before the real board. A model this small
# follows a pattern it has just seen far more reliably than a written instruction.
EXAMPLES = [
    (
        [{"defect": "missing_hole", "severity": "High", "location": "top-left"},
         {"defect": "mouse_bite", "severity": "Medium", "location": "centre"}],
        "Action: Hold this board. First compare the missing hole at top-left with the drill file and re-drill it "
        "if the pad allows, otherwise reject the board. Then measure the remaining track width at the mouse bite "
        "in the centre and repair or reject it if it is below the design minimum.",
    ),
    (
        [{"defect": "spurious_copper", "severity": "Low", "location": "top-right"},
         {"defect": "spurious_copper", "severity": "Low", "location": "top-right"},
         {"defect": "spurious_copper", "severity": "Low", "location": "bottom-centre"}],
        "Action: Hold this board. Check the three patches of spurious copper at top-right and bottom-centre, "
        "remove any that are close to a track or pad, then record the rest and accept the board.",
    ),
]
ANSWER_PREFIX = "Action:"        # the answer is forced to begin with this word


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


def group_findings(findings):
    """Groups defects of the same type and severity, most severe first.

    Returns a list of (defect, severity, count, places), where places are the
    distinct location names in the order they were found.
    """
    groups = {}
    for f in findings:
        places = groups.setdefault((f["defect"], f["severity"]), {"count": 0, "places": []})
        places["count"] += 1
        if f["location"] not in places["places"]:
            places["places"].append(f["location"])
    ordered = sorted(groups.items(), key=lambda item: LEVELS.index(item[0][1]), reverse=True)
    return [(defect, severity, g["count"], g["places"]) for (defect, severity), g in ordered]


def group_by_type(findings):
    """Groups defects by type only, for the operator instruction.

    One defect type needs one action however its members were graded, so all of
    its locations go on one line. Returns a list of (defect, highest severity,
    count, places), most severe type first, most severe locations first.
    """
    groups = {}
    for defect, severity, count, places in group_findings(findings):      # already most severe first
        g = groups.setdefault(defect, {"severity": severity, "count": 0, "places": []})
        g["count"] += count
        g["places"] += [p for p in places if p not in g["places"]]
    return [(defect, g["severity"], g["count"], g["places"]) for defect, g in groups.items()]


def _join(words):
    """['a', 'b', 'c'] -> 'a, b and c'"""
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def summary_line(findings):
    """The factual summary, written by code so that it is always exact."""
    if not findings:
        return "No surface defects detected."
    parts = [f"{count} {defect} ({severity}) at {_join(places)}" for defect, severity, count, places in group_findings(findings)]
    return f"{len(findings)} defect(s): " + "; ".join(parts) + "."


def build_user_prompt(findings):
    """Writes each defect type on one line together with its locations and standard action."""
    lines = []
    for defect, severity, count, places in group_by_type(findings):
        action = DEFECT_ACTIONS.get(defect, UNKNOWN_ACTION)
        lines.append(f"- {count} {defect}, severity {severity}, at {_join(places)}. Standard action: {action}.")
    return "\n".join(lines)


def rule_based_action(findings):
    """The operator instruction written without a language model."""
    sentences = ["Hold this board."]
    for defect, _, count, places in group_by_type(findings):
        action = DEFECT_ACTIONS.get(defect, UNKNOWN_ACTION)
        what = f"the {defect}" if count == 1 else f"the {count} {defect} defects"
        sentences.append(f"For {what} at {_join(places)}, {action}.")
    return "Action: " + " ".join(sentences)


def check_action(answer, findings):
    """Tidies Llama's instruction and checks it against the findings.

    Returns the instruction, or raises RuntimeError saying what was wrong with it.
    """
    lines = answer.strip().splitlines()
    text = lines[0].strip() if lines else ""                    # ignore anything after the first paragraph
    if not text.startswith(ANSWER_PREFIX):
        raise RuntimeError("Llama's answer did not start with Action:")
    if text[-1] not in ".!":                                    # cut off at the length limit:
        text = text[:text.rfind(".") + 1]                       # keep only the complete sentences
    if len(text) < len(ANSWER_PREFIX) + 20:
        raise RuntimeError("Llama's answer was empty or cut off")

    found_types = {f["defect"] for f in findings}
    mentioned = {name for name, pattern in DEFECT_PATTERNS.items() if pattern.search(text)}
    # Every defect type that was found must be named, so nothing is silently skipped...
    groups = group_by_type(findings)
    left_out = [d for d, _, _, _ in groups if d in DEFECT_PATTERNS and d not in mentioned]
    if left_out:
        raise RuntimeError(f"Llama's answer left out a defect that was found ({', '.join(left_out)})")
    # ...every location must be named too, so the operator knows where to look
    # ("top-right" may be written "top right")...
    all_places = [p for _, _, _, places in groups for p in places]
    missing_places = [p for p in dict.fromkeys(all_places)
                      if not re.search(r"\b" + p.replace("-", "[ -]?") + r"\b", text, re.IGNORECASE)]
    if missing_places:
        raise RuntimeError(f"Llama's answer left out a location ({', '.join(missing_places)})")
    # ...and no defect type may be named that was not found.
    invented = sorted(mentioned - found_types)
    if invented:
        raise RuntimeError(f"Llama's answer named a defect that was not found ({', '.join(invented)})")
    return text


def llama_available():
    """True when the ExecuTorch environment, the runner script and the model file all exist."""
    return all(os.path.exists(p) for p in (ET_PYTHON, RUNNER, MODEL_FILE))


def ask_llama(user_text, max_new_tokens=120):
    """Runs llama_runner.py in the ExecuTorch environment and returns its result dictionary.

    The runner is a separate program, so we send the request as JSON on its standard
    input and look for the line it prints that starts with RESULT_JSON:.
    Raises RuntimeError if anything goes wrong.
    """
    request = json.dumps({
        "system": SYSTEM_PROMPT,
        "examples": [[build_user_prompt(facts), answer] for facts, answer in EXAMPLES],
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
    """Turns the findings into what the operator sees.

    Returns a dictionary:
      summary : the factual line, always written by code
      text    : the instruction for the operator
      source  : 'llama' (written on-device by Llama via ExecuTorch), 'rules'
                (written by the fixed rules) or 'none' (clean board)
      stats   : speed figures from Llama, when it was used
      raw     : Llama's full answer before tidying, when it produced one
      error   : why Llama's answer was not used, when that happened
    """
    summary = summary_line(findings)
    if not findings:
        return {"summary": summary, "text": "Action: Pass the board to the next stage.", "source": "none", "stats": {}}

    fallback = {"summary": summary, "text": rule_based_action(findings), "source": "rules", "stats": {}}
    if not llama_available():
        fallback["error"] = "Llama is not installed on this computer"
        return fallback

    try:
        result = ask_llama(build_user_prompt(findings))
    except Exception as exc:                      # timeout, crash, unreadable output
        fallback["error"] = str(exc)
        return fallback
    try:
        text = check_action(result["text"], findings)
    except RuntimeError as exc:                   # an answer that failed the checks
        fallback.update(error=str(exc), raw=result["text"], stats=result.get("stats", {}))
        return fallback
    return {"summary": summary, "text": text, "source": "llama", "stats": result.get("stats", {}), "raw": result["text"]}


if __name__ == "__main__":
    # Quick test without the dashboard:  python3 llm_engine.py
    # Four different boards, to see how Llama copes with each.
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
        "D: three spurious copper patches, one small and two large": [
            {"defect": "spurious_copper", "confidence": 0.74, "box": [2300, 300, 2340, 388]},
            {"defect": "spurious_copper", "confidence": 0.75, "box": [2400, 700, 2440, 803]},
            {"defect": "spurious_copper", "confidence": 0.78, "box": [1500, 780, 1621, 810]},
        ],
    }
    for name, detections in cases.items():
        case_findings = add_severity_and_location(detections, *SIZE)
        print(f"\n================ Case {name} ================")
        print(build_user_prompt(case_findings))
        answer = advise(case_findings)
        print(f"\n--- Shown to the operator (instruction source: {answer['source']}) ---")
        print(answer["summary"])
        print(answer["text"])
        if answer.get("raw"):
            print("\n--- Llama's full answer before tidying ---")
            print(answer["raw"])
        if answer.get("error"):
            print("\nWhy Llama's answer was not used:", answer["error"])
        if answer["stats"]:
            print("\n--- Speed ---")
            for key, value in answer["stats"].items():
                print(f"{key:32s} {value}")
