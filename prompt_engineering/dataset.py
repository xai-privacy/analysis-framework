"""Dataset loading and the dev/test split.

The split is by exam year: 2021-2023 for iteration (60 problems), 2024-2025 held
out for final numbers (37 problems). Testing many prompting strategies and
reporting the best one on the same 97 problems would give an optimistically
biased result, since the choice of strategy is itself fit to the data.

Few-shot demonstrations are always drawn from dev, never test.
"""

import json
from pathlib import Path
from typing import List

DATA_PATH = Path(__file__).parent / "data" / "LEET_Arg_Questions.json"

DEV_YEARS = {"2021", "2022", "2023"}
TEST_YEARS = {"2024", "2025"}


def load_all(path: Path = DATA_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_split(split: str, path: Path = DATA_PATH) -> List[dict]:
    qs = load_all(path)
    if split == "dev":
        return [q for q in qs if q["year"] in DEV_YEARS]
    if split == "test":
        return [q for q in qs if q["year"] in TEST_YEARS]
    if split == "all":
        return qs
    raise ValueError(f"unknown split '{split}' (use dev, test, or all)")


def example_pool(path: Path = DATA_PATH) -> List[dict]:
    """Few-shot demonstrations: dev only, and only items with a usable rationale."""
    return [q for q in get_split("dev", path) if (q.get("original_rationale") or "").strip()]
