# LEET-Arg (cleaned)

Cleaned copy of the LEET-Arg benchmark used by the analysis framework. This directory holds a
**derived artifact**, not an original dataset. The upstream source is authoritative for anything
not listed under "Changes applied" below.

## Provenance

| | |
|---|---|
| Upstream repo | `lit-ai-lab/leet_arg_dataset_v1` |
| Upstream file | `LEET_Arg_Questions.json` |
| Upstream commit | `599ef7a80a32422f5d5f3bd718df12e6c4042bf7` <!-- TODO: confirm this is the commit you actually pulled from; this was HEAD on 2026-08-17 --> |
| Cleaned on | <!-- TODO: date you ran the script --> |
| Produced by | `tools/clean_leet_arg.py` (this repo), **plus one manual edit**, see below |
| Verified by | `tools/validate_leet_arg.py` (this repo) |
| Source paper | Park and Park, "When correct is not enough," https://doi.org/10.1016/j.knosys.2026.116625 |
| sha256 of `leet_arg_clean_v1.json` | `c3607b34ba5dc669b2e10283af9bfc00ef05fec32c350e683e20d165ab3dd77d` |

The script alone does **not** reproduce the shipped file. Running

```bash
python tools/clean_leet_arg.py \
  --input  data/leet_arg/LEET_Arg_Questions.json \
  --output data/leet_arg/leet_arg_clean_v1.json
```

yields sha256 `ac209e84d2c92ddcee589c5a42576448c0470eba5d3af01fd49b361e1543a88a`. The shipped file is
that output plus the manual `2022_31` fix described under "Changes applied", which was applied by
hand downstream of the script. Anyone regenerating from scratch has to re-apply that edit or the
result will not match the hash above. Folding it into the script is a `_v2` task.

## Contents

`leet_arg_clean_v1.json` is a JSON list of 97 records totalling 315 statement units, which matches
the counts reported in the source paper.

Each record has: `id`, `year`, `problem_idx`, `objective`, `domain`, `category`, `answer`,
`original_question`, `statements`, `original_rationale`.

Notes for anyone consuming the file:

- `answer` is a **string** ("1" through "5"), not an integer.
- `statements` has 3 entries for 85 records and 5 entries for 12 records. In the 3-statement shape
  the choices combine labelled sub-statements; in the 5-statement shape each choice is itself a
  statement. The 5-statement records are `2021_01, 2021_03, 2021_25, 2022_36, 2023_08, 2023_19,
  2023_22, 2023_23, 2024_05, 2024_14, 2024_25, 2025_21`.
- `domain` is `null` for 10 records: `2021_36, 2021_37, 2022_38, 2022_39, 2023_35, 2023_37,
  2024_37, 2025_26, 2025_39, 2025_40`.
- Answer distribution is 1:24, 2:25, 3:20, 4:14, 5:14, so a majority-class baseline is 25/97.

## Changes applied

`statements` was modified throughout. `original_question` was modified for exactly one record,
`2022_31`, described below. `original_rationale`, `answer`, and all metadata fields
(`id`, `year`, `problem_idx`, `objective`, `domain`, `category`) are byte-identical to upstream for
all 97 records, verified field by field against `LEET_Arg_Questions.json`.

**Segmentation rule.** Statements are split on the *next expected label only*. After `(a)` the
only thing that can begin the next statement is `(b)`; after `①` only `②`. This is what prevents
false splits on things like `(B)` mid-sentence, `Group 2.`, `$10.`, `1.5% to 3.5%`, and
`chromosomes (n)`. Two label schemes are recognised, circled numerals and `(a)` through `(e)`.

**Re-segmented from `original_question` (9 records):**
`2021_25, 2021_34, 2022_19, 2023_11, 2023_17, 2023_29, 2024_27, 2025_17, 2025_39`

These had confirmed segmentation errors upstream. `2021_25` is the case where the statements sit
under `<choices>` rather than `<statements>`, so the script falls back to reading the choices block.

**Manually reconstructed (1 record):** `2025_05`

Upstream had `statements: null` for this record and the text sits in an unusual position, before
`<question>` rather than after `<statements>`. Because of that, it wasn't worth extending the
segmentation rule for a single record, so Mohammed Karim reconstructed the three statements by
hand when `tools/clean_leet_arg.py` was first written. They're hardcoded in
`MANUAL_2025_05_STATEMENTS` in the cleaning script rather than derived, so any change to this
record has to be made in the script, not in the JSON.

`original_question` for `2025_05` itself is untouched, byte-identical upstream text (the only
other-field edit anywhere in the corpus is the `2022_31` fix below). On 2026-08-21 this text was
used to fix `benchmarks/LEET_Arg_Questions_cleaned.json`'s copy of the same record, which had
picked up duplicated statement text from whatever source that separate file was assembled from.

**Choice-marker scheme normalised in `original_question` (1 record):** `2022_31`

The `<choices>` block of `2022_31` used plain `1.` through `5.` where every other record in the
corpus uses circled numerals. The five markers were replaced with `①` through `⑤`; the diff against
upstream is exactly those five substitutions and nothing else. This is the only edit anywhere in the
corpus to a field other than `statements`.

Two things to know about this one:

- It was applied **by hand, not by `tools/clean_leet_arg.py`**. The script has no rule for it, so a
  clean re-run reverts the record to plain numbering. See "Provenance" for the two hashes.
- It is easy to find this change misattributed to `2025_05` in the issue thread that produced the
  file. That attribution is wrong; `2025_05` was untouched by this fix, and its own change is the
  manual reconstruction described above. Verified by diffing the pre- and post-fix files: `2022_31`
  is the only record that differs between them.

**Cosmetic label spacing normalised (7 records):**
`2021_13, 2022_08, 2023_29, 2024_06, 2024_22, 2024_29, 2025_23`

`"(a) . If ..."` becomes `"(a). If ..."`. Content is unchanged.

**Deliberately left alone (5 records):**
`2022_36, 2023_19, 2024_25, 2025_21, 2025_01`

These trip the linter but are correct as they stand. They are recorded as known false positives so
that a future pass does not "fix" them. `2022_36` in particular contains Graphviz `digraph` source
inside its statements, which looks like a parse failure and is not one.

## Known limitations

The linter compares each record against the dominant formatting convention, so it cannot flag a
record that deviates from that convention without a reference. `2022_31` was the known example: it
used plain numbering where the rest of the corpus uses circled Unicode characters, and no
rule-based check caught it. It was found by a human read and fixed by hand in the shipped file, but
the underlying gap is unchanged, so anomalies of that kind still need a human read. The fix also
lives only in the JSON, not in the cleaning script.

The script asserts a total of 315 statement units at the end of the run and warns if the count
differs. That check catches gross breakage but will not catch a statement that was split at the
wrong boundary while preserving the count.

`2023_29` appears in both the auto-fix and cosmetic lists. The auto-fix path runs first and returns
early, but it also normalises spacing, so the outcome is the same either way. The overlap is
harmless and noted here only so it does not read as a bug.

## Versioning

The filename carries the version. Do not overwrite `leet_arg_clean_v1.json` in place. Any change to
the cleaning logic gets a new `_v2` file and a new section in this README, because every results
file needs to name the exact dataset snapshot that produced its numbers.

## Related

- Issue #14, dataset cleaning, for the full findings and discussion
- Issue #13, analysis framework, for how this file is consumed
