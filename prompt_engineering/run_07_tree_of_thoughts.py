#!/usr/bin/env python3
"""Technique 7/8 -- Tree-of-Thoughts (SINGLE-LEVEL REDUCTION).

Branch into k candidate lines of attack, have the model score them, then expand
the best one into a full solution.

!! This is NOT the published ToT algorithm. Real ToT searches a multi-level thought
!! tree with backtracking. That is impractical against a 3B model on CPU, and a
!! single 5-way multiple-choice judgement offers little to backtrack over. What runs
!! here keeps ToT's characteristic shape -- branch, score, expand -- at one level
!! with beam width 1. Report results as evidence about this reduction. If it shows
!! promise, that is an argument for implementing full ToT, not a substitute for it.

    python run_07_tree_of_thoughts.py --limit 10 --k-branches 3

Cost: k + 2 model calls per question (default 5).
"""

from common import build_parser, execute


def add_args(ap):
    ap.add_argument("--k-branches", type=int, default=3,
                    help="candidate lines of reasoning to generate (default 3)")


if __name__ == "__main__":
    execute("tree_of_thoughts", build_parser(__doc__, add_args).parse_args())
