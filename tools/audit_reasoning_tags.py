# File: tools/audit_reasoning_tags.py
"""Scan existing results/*.json files for undeclared reasoning tags.

Retroactive check for the same thing run_benchmark.py now warns about live:
a response opening with a paired open/close tag (e.g. <think>...</think> or
[THINK]...[/THINK]) that isn't declared in that model's model_configs/*.json
"reasoning" field. Catches cases like Ministral's [THINK] syntax, which the
parser's <think>-only scoping would otherwise silently miss.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_benchmark import detect_undeclared_reasoning_tag

_RESULTS_GLOB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "*.json")
_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_configs")


def _declared_open_tag(signature):
    config_path = os.path.join(_CONFIGS_DIR, f"{signature}.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return (config.get("reasoning") or {}).get("open_tag")


def main():
    any_flagged = False
    for path in sorted(glob.glob(_RESULTS_GLOB)):
        signature = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        declared = _declared_open_tag(signature)
        flagged_count = 0
        for record in records:
            raw = record.get("model_rationale") or ""
            flagged = detect_undeclared_reasoning_tag(raw, declared)
            if flagged:
                flagged_count += 1
                any_flagged = True
                print(f"{path} [{record.get('id')}]: opens with {flagged!r} (declared: {declared!r})")

        print(f"{path}: {flagged_count}/{len(records)} flagged\n")

    if not any_flagged:
        print("No undeclared reasoning tags found in any results/*.json file.")


if __name__ == "__main__":
    main()
