#!/usr/bin/env python3
"""Technique 1/8 -- Zero-shot (BASELINE).

The paper's own prompt: state the problem, ask for a choice and explanations.
No reasoning scaffold of any kind. Every other technique is measured against this,
so run it first.

    python run_01_zero_shot.py --limit 10
    python run_01_zero_shot.py --runs 3

Cost: 1 model call per question.
"""

from common import build_parser, execute

if __name__ == "__main__":
    execute("zero_shot", build_parser(__doc__).parse_args())
