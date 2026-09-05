#!/usr/bin/env python3
"""Technique 4/8 -- Self-consistency.

Samples k independent CoT paths at non-zero temperature and takes the majority
answer. Ties break toward the earliest sample.

    python run_04_self_consistency.py --limit 10 --k-samples 5

Cost: k model calls per question (default 5). This is the first expensive
technique -- expect roughly 5x the wall clock of zero-shot.

Interpretation note: self-consistency reduces run-to-run variance by construction,
so it should look strong on exactly the axis where small models are weakest. Judge
it against its compute cost, which is why tokens/question is reported.
"""

from common import build_parser, execute


def add_args(ap):
    ap.add_argument("--k-samples", type=int, default=5,
                    help="number of sampled reasoning paths to vote over (default 5)")
    ap.add_argument("--sc-temperature", type=float, default=0.7,
                    help="sampling temperature for the paths (default 0.7; needs > 0)")


if __name__ == "__main__":
    execute("self_consistency", build_parser(__doc__, add_args).parse_args())
