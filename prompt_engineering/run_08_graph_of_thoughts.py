#!/usr/bin/env python3
"""Technique 8/8 -- Graph-of-Thoughts (SINGLE-LEVEL REDUCTION).

Generate k independent partial analyses, aggregate them into one coherent analysis
(resolving contradictions explicitly), then refine that into a final decision.

!! This is NOT the published GoT algorithm. Real GoT maintains a scored graph of
!! thoughts with multi-round aggregation and refinement. What runs here preserves
!! GoT's defining operation -- aggregation of several thoughts into a stronger one,
!! as opposed to ToT's pure branching -- but drops the scored multi-round graph.
!! Report accordingly.

    python run_08_graph_of_thoughts.py --limit 10 --k-thoughts 3

Cost: k + 2 model calls per question (default 5).
"""

from common import build_parser, execute


def add_args(ap):
    ap.add_argument("--k-thoughts", type=int, default=3,
                    help="partial analyses to generate before aggregating (default 3)")


if __name__ == "__main__":
    execute("graph_of_thoughts", build_parser(__doc__, add_args).parse_args())
