"""Advisory pipeline: Llama 3.2 1B Instruct running on ExecuTorch (XNNPACK, Arm CPU).

This script must be run with the Python of the ExecuTorch environment (~/et_env),
not the dashboard's environment. It has nothing to do with the vision model.

Two ways to use it:

  1. Quick test from the terminal (prints the answer as it is generated, then the speed):
         ~/et_env/bin/python llama_runner.py --test

  2. Called by the dashboard (llm_engine.py): it reads a JSON request from standard
     input and prints one line starting with RESULT_JSON: that holds the answer.
"""
import argparse
import json
import os
import resource
import sys
import time

MODEL_DIR = os.path.expanduser("~/llama")
DEFAULT_MODEL = "llama3_2_1b_main.pte"
DEFAULT_TOKENIZER = "tokenizer.model"
RESULT_MARKER = "RESULT_JSON:"

TEST_SYSTEM = (
    "You are a quality-control assistant on a PCB manufacturing line. "
    "Answer in at most three short sentences."
)
TEST_USER = (
    "Inspection findings: 2 defects. "
    "1) short, severity High, location top-right, confidence 0.91. "
    "2) spur, severity Medium, location centre, confidence 0.72. "
    "Explain the likely cause and tell the operator what to do."
)


def build_chat_prompt(system_text, user_text):
    """Wraps the text in the special markers Llama 3 Instruct models were trained with.

    Without these markers the model does not know where the instructions end and
    where its own answer should begin.
    """
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_text}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate(system_text, user_text, model_path, tokenizer_path, max_new_tokens=160, stream=False):
    """Loads the model, generates one answer and returns it with timing numbers."""
    # These two imports register extra operators that the exported model uses
    # (the fast attention operator and the 4-bit lookup table).
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    from executorch.kernels import quantized  # noqa: F401
    from executorch.extension.llm.runner import GenerationConfig, TextLLMRunner

    t_start = time.perf_counter()
    runner = TextLLMRunner(model_path, tokenizer_path)

    config = GenerationConfig(
        echo=False,                    # do not repeat our prompt back in the output
        max_new_tokens=max_new_tokens, # upper limit on the length of the answer
        temperature=0.0,               # 0 = always pick the most likely word (repeatable answers)
    )

    pieces = []
    stats = {}

    def on_token(text):
        pieces.append(text)
        if stream:
            print(text, end="", flush=True)

    def on_stats(s):
        # Copy the numbers out now; the object is only valid during this call.
        stats["prompt_tokens"] = s.num_prompt_tokens
        stats["generated_tokens"] = s.num_generated_tokens
        stats["time_to_first_token_ms"] = s.first_token_ms - s.inference_start_ms
        stats["generation_ms"] = s.inference_end_ms - s.first_token_ms

    runner.generate(build_chat_prompt(system_text, user_text), config, on_token, on_stats)
    total_ms = (time.perf_counter() - t_start) * 1000

    text = "".join(pieces)
    for marker in ("<|eot_id|>", "<|end_of_text|>"):
        text = text.replace(marker, "")

    gen_ms = stats.get("generation_ms", 0)
    generated = stats.get("generated_tokens", 0)
    stats["tokens_per_second"] = round(generated * 1000 / gen_ms, 2) if gen_ms > 0 else None
    stats["total_ms_including_model_load"] = round(total_ms)
    # ru_maxrss is the most memory this process has used, in kilobytes on Linux.
    stats["peak_memory_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return {"text": text.strip(), "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Run Llama 3.2 1B on ExecuTorch")
    parser.add_argument("--test", action="store_true", help="run a built-in PCB example and print the speed")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model file name inside ~/llama")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="tokenizer file name inside ~/llama")
    args = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, args.model)
    tokenizer_path = os.path.join(MODEL_DIR, args.tokenizer)
    for path in (model_path, tokenizer_path):
        if not os.path.exists(path):
            sys.exit(f"File not found: {path}")

    if args.test:
        print(f"Model: {model_path}\nTokenizer: {tokenizer_path}\n\n--- Answer ---")
        result = generate(TEST_SYSTEM, TEST_USER, model_path, tokenizer_path, stream=True)
        print("\n\n--- Speed ---")
        for key, value in result["stats"].items():
            print(f"{key:32s} {value}")
        return

    request = json.loads(sys.stdin.read())
    result = generate(
        request["system"], request["user"], model_path, tokenizer_path,
        max_new_tokens=int(request.get("max_new_tokens", 160)),
    )
    print(RESULT_MARKER + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
