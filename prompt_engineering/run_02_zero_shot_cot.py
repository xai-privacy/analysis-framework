#!/usr/bin/env python3
"""Technique 2/8 -- Zero-shot chain-of-thought.

Adds "let's think step by step" and asks the model to walk each option in turn
before committing. This is the cheapest possible CoT: no examples, no extra calls,
just a reasoning instruction.

    python run_02_zero_shot_cot.py --limit 10

Cost: 1 model call per question (but longer outputs than zero-shot).

Note: CoT gains were originally emergent with model scale and can *degrade* small
models. A drop relative to the baseline here is a real finding, not a bug.
"""

from common import build_parser, execute

if __name__ == "__main__":
    execute("zero_shot_cot", build_parser(__doc__).parse_args())
