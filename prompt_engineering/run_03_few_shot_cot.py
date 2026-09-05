#!/usr/bin/env python3
"""Technique 3/8 -- Few-shot chain-of-thought.

Prepends k worked examples, using the dataset's own expert-authored rationales as
the demonstration reasoning. Those are gold-standard legal reasoning, which makes
them unusually strong demonstrations.

Leakage guard: examples are drawn only from the dev split, and the current question
is always excluded from its own prompt.

    python run_03_few_shot_cot.py --limit 10 --k-examples 2

Cost: 1 model call per question, but a much longer prompt -- watch tokens/question.
Raising --k-examples can overflow a small model's context window.
"""

from common import build_parser, execute


def add_args(ap):
    ap.add_argument("--k-examples", type=int, default=2,
                    help="number of worked examples to prepend (default 2)")


if __name__ == "__main__":
    execute("few_shot_cot", build_parser(__doc__, add_args).parse_args())
