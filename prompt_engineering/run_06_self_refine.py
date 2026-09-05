#!/usr/bin/env python3
"""Technique 6/8 -- Self-refine.

Three stages: answer, critique that answer, then revise. The critique stage is
told to look for unsupported premises, steps that don't follow, options dismissed
without reason, and reasoning that merely restates the question.

Aimed squarely at the paper's headline finding -- correct answers supported by
invalid reasoning.

    python run_06_self_refine.py --limit 10

Cost: 3 model calls per question.

What to check in the traces: sycophantic collapse, where the critique just agrees
with the original attempt and the revision changes nothing. All three stages are
saved, so run `python score.py --detail self_refine --show-trace 1` to look.
"""

from common import build_parser, execute

if __name__ == "__main__":
    execute("self_refine", build_parser(__doc__).parse_args())
