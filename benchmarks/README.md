# LEET-Arg (cleaned)

`LEET_Arg_Questions_cleaned.json` is a cleaned copy of the LEET-Arg benchmark (Park and Park, "When
correct is not enough," https://doi.org/10.1016/j.knosys.2026.116625; upstream repo
`lit-ai-lab/leet_arg_dataset_v1`, file `LEET_Arg_Questions.json`).
sha256: `f52a9a8f0405192a53990e56eef04ae0fd0d474f049ac6e721255332f8fc3cea`

`LEET_Arg_Model_Responses.json` in this directory holds per-model responses keyed by the same
`id`/`year`/`problem_idx` and is out of scope here.

## Contents

93 records, 301 statement units (upstream has 97 / 315 — see "Changes applied" for why). Each record
has: `id`, `year`, `problem_idx`, `objective`, `domain`, `category`, `answer`, `original_question`,
`statements`, `original_rationale`.

- `answer` is a **string** ("1" through "5"), not an integer.
- `statements` has 3 entries for 82 records and 5 entries for 11: 2021_03, 2021_25, 2022_36, 2023_08,
  2023_19, 2023_22, 2023_23, 2024_05, 2024_14, 2024_25, 2025_21.
- `domain` is `null` for 9 records: 2021_36, 2021_37, 2022_38, 2022_39, 2023_35, 2024_37, 2025_26,
  2025_39, 2025_40.
- Answer distribution: 1:24, 2:24, 3:19, 4:12, 5:14 (majority-class baseline 24/93).

## Changes applied

Produced by `tools/clean_leet_arg.py`, plus manual edits. `original_rationale`, `answer`, and all
metadata fields are byte-identical to upstream; everything below touches `statements` and/or
`original_question`.

- **Statements re-derived** from `original_question` for 9 records with confirmed upstream
  segmentation errors, using a "next expected label only" rule: 2021_25, 2021_34, 2022_19, 2023_11,
  2023_17, 2023_29, 2024_27, 2025_17, 2025_39. All other records' `statements` are upstream text.
- **2025_05**: statements manually reconstructed (upstream had `statements: null`). Hardcoded as
  `MANUAL_2025_05_STATEMENTS` in the script, so future edits go there, not the JSON.
- **2022_31**: `original_question` choice markers normalised from 1.–5. to ①–⑤. Applied by hand, not
  by the script — a clean re-run reverts it.
- **4 records dropped** for inconsistent `original_rationale`, not repaired: 2021_01, 2021_13,
  2021_15, 2023_37.
- **Cosmetic spacing** normalised (`"(a) ."` → `"(a)."`) in 6 records: 2022_08, 2023_29, 2024_06,
  2024_22, 2024_29, 2025_23.

See issue #14 for the full cleaning discussion.
