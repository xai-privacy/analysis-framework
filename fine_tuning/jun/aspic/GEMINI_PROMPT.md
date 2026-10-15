# Gemini prompt for drafting ASPIC+ translations

All training translations come from Gemini, so the student learns from one teacher. The five translations in `translations_claude/` were written by Claude during the pilot. They are used only as the first worked examples and as a comparison, never as training data.

Run everything from `fine_tuning/jun/aspic/` with the venv active (`source .venv/bin/activate`, see README.md). Below, `B` is the cleaned benchmark file and `S` is your split:

```bash
B=../../../benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json
S=../data/split.json
```

## Round A: Gemini redoes the three example questions

They are shown two other examples, so Gemini cannot copy its own answer.

```bash
python make_gemini_batch.py --bench $B --split $S \
    --examples-dir translations_claude --examples 2022_29 2023_26 \
    --ids 2022_17 2021_37 2024_15 --out gemini_input_A.md
```

In Gemini (Pro model), start a new chat, attach `gemini_input_A.md`, and paste the prompt below. Save the JSON array from the answer as `gemini_output_A.json`, then:

```bash
python check_aspic.py --bench $B --split $S --save-matches translations gemini_output_A.json
```

From now on, examples are taken from Gemini's own translations in `translations/`.

## Round B: the other two pilot questions plus three new ones

```bash
python make_gemini_batch.py --bench $B --split $S \
    --ids 2022_29 2023_26 2021_28 2021_34 2023_22 --out gemini_input_B.md
```

Same steps: new Gemini chat, attach, paste the prompt, save as `gemini_output_B.json`, run `check_aspic.py ... --save-matches translations gemini_output_B.json`.

## Later rounds

```bash
python make_gemini_batch.py --bench $B --split $S --next 5 --out gemini_input_C.md
```

Notes:
- Start a new Gemini chat for every round, so earlier answers do not leak into later ones.
- Only translations that reproduce the gold answer are saved. For a WRONG or UNRESOLVED one, tell Gemini what came out wrong (in the same chat) and check again, or drop the question.
- Read each saved translation once against its passage before training.
- Keep the input files, Gemini's raw outputs and the date of each round in the repo. A chat model will not produce the same output twice.

## Prompt

```
I attached a file with three parts: (1) a JSON format for translating LEET argument questions into ASPIC+ argument structure, (2) worked examples with finished translations, and (3) the questions I need translated.

Translate every question in Part 3 into the format in Part 1, in the same style as the examples in Part 2.

Rules:
1. Build the structure from the passage. Use the expert rationale to understand the reasoning. Anything you take from the rationale gets "source": "rationale", and general knowledge stated nowhere gets "source": "background".
2. Do not write verdicts anywhere: no "correct", "incorrect", "true", "false", and no answer number. Whether a statement is true must follow only from the structure and its check.
3. When the checks are evaluated as described in Part 1, each check must give the same result as the expert rationale. Do not add a rule, contrary or premise only to make a check come out right. Every element must be justified by the passage or the rationale.
4. If a statement is false because a tempting inference fails, write that inference and the reason it fails (often an undercut), as in statement (c) of 2021_37 if that example is included.
5. Keep claim text close to the passage wording. Do not invent speakers or facts. Use the position labels the passage uses.
6. Single-choice questions (the choices are the statements themselves) use the keys "1" to "5" in "statements".
7. If a statement cannot be expressed with the check operations in Part 1, use {"op": "unsupported", "reason": "..."} instead of forcing it.
8. You may add a short "note" to a statement to explain a choice that is not obvious.
9. Before answering, check that every claim ID, rule ID, position label and hypothetical name you reference is defined.

Output: one JSON code block containing an array with one object per question, in the order of Part 3. After the code block, list in plain text every statement you were unsure about, and why.
```
