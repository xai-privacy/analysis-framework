#!/usr/bin/env python3
"""Technique 5/8 -- Plan-and-solve.

Asks the model to first devise a plan (identify the competing positions, the
premises each depends on, what would have to be true for each option), then carry
it out while stating every premise it relies on.

Targets CoT's characteristic missing-step failure -- which is exactly what the
LEET-Arg paper documents: "logical leaps between premises and conclusions,
omission of necessary premises".

    python run_05_plan_and_solve.py --limit 10

Cost: 1 model call per question.
"""

from common import build_parser, execute

if __name__ == "__main__":
    execute("plan_and_solve", build_parser(__doc__).parse_args())
