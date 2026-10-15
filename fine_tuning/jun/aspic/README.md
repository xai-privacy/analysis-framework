# ASPIC+ translation pilot

Goal: check whether LEET-Arg questions can be written in an ASPIC+ format from which a solver, not the model, computes the answer. If yes, the next step is to translate the train and validation questions and fine-tune Llama 3.2 3B to produce this format (same split and settings as the first run).

Put this folder at `fine_tuning/jun/aspic/`.

| File | What it is |
|---|---|
| `ASPIC_FORMAT.md` | The format: fields, check operations, writing rules, and the frozen instruction for training |
| `check_aspic.py` | Computes each statement from a translation and compares the result with the gold answer. No model involved |
| `translations_claude/*.json` | The five pilot translations written by Claude (2021_37, 2022_17, 2022_29, 2023_26, 2024_15). First worked examples and a comparison only, never training data |
| `translations/` | Empty at first. Gemini's translations that pass the check are saved here, one file per question. This is the training data |
| `make_gemini_batch.py` | Builds the single file to attach to Gemini (format, 3 examples, questions with rationales) |
| `GEMINI_PROMPT.md` | The rounds to run, and the prompt to paste into Gemini |
| `requirements.txt`, `.gitignore` | PyArg version for the venv; keeps `.venv/` out of git |

## Environment (venv)

The scripts need Python 3.9 or newer. The only package is PyArg (`python-argumentation`, pinned in `requirements.txt`). Without it the built-in solver is used and the results are the same.

One-time setup:

```bash
cd fine_tuning/jun/aspic
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In every new terminal:

```bash
cd fine_tuning/jun/aspic
source .venv/bin/activate
```

`.gitignore` keeps `.venv/` out of commits.

## Run the check

```bash
python check_aspic.py \
    --bench ../../../benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
    --split ../data/split.json -v translations_claude/
```

The script refuses the 16 final test questions and warns about validation questions. When PyArg is installed, it computes the grounded extension with PyArg and checks that the built-in solver agrees.

## Pilot result

| Question | Category | Statement types | Statements | Answer | Gold |
|---|---|---|---|---|---|
| 2021_37 | Argument Evaluation & Problem Solving | strengthens, weakens | F T F | ② | ② |
| 2022_17 | Argumentation and Rebuttal | holds, assumes | T T T | ⑤ | ⑤ |
| 2022_29 | Argument Evaluation & Problem Solving | strengthens, weakens | T F T | ③ | ③ |
| 2023_26 | Argument Analysis | weakens, assumes, holds | T F F | ① | ① |
| 2024_15 | Argument Evaluation & Analysis | status under a criterion, holds | T T T | ⑤ | ⑤ |

All 5 reproduce the gold answer, and no statement had to be marked unsupported. To make sure the checks are not trivially true, I broke each translation on purpose (removed an undercut, a premise, made a criterion defeasible, and so on). Each time the expected statement flipped and the answer became wrong.

### Observations

1. **40 of 102 claims and rules come from the expert rationale, not the passage.** At test time the model only sees the passage, so it has to supply these itself. The share is highest for Argument Analysis (2023_26: 12 of 18), where the statements are about assumptions the passage never states.
2. **For weakens/strengthens (7 of 15 statements), the translation decides the answer.** Whether the solver says "weakened" depends on whether the translation includes the rule that connects the evidence to the claim. The solver does real work only where there is a counter-argument: in 2022_29 (b) and 2021_37 (c), the tempting attack is defeated by an undercut (the 20-year lag; caloric is massless), and the solver computes that the claim survives.
3. **One format change was needed:** named hypotheticals, because 2024_15 reuses the same "if criterion K" block in two statements.
4. **Interpretation call:** 2023_26 (a) "method A is difficult to apply" is translated as "A's assumption is undermined, so A is weakened".

## Caveats

- I wrote the translations from the public LEET-Arg text. Run the check once against the cleaned benchmark file to confirm they still match.
- I rebuilt your seed-42 split to pick train questions. The 15 validation IDs match your Oct 3 run, but `--split ../data/split.json` is the authority.

## Gemini workflow (one teacher for all training data)

`GEMINI_PROMPT.md` has the exact commands. In short:

1. **Round A:** Gemini redoes 2022_17, 2021_37 and 2024_15. It is shown 2022_29 and 2023_26 as examples, so it cannot copy the answers it is translating.
2. **Round B:** Gemini redoes 2022_29 and 2023_26 plus three new questions, with its own round-A translations as examples.
3. **Later rounds:** `--next 5` each time.

After each round, `check_aspic.py --save-matches translations <gemini output>` saves the translations that reproduce the gold answer. Read each saved one once against its passage. Claims marked `passage` that are not in the text show up as warnings.

Comparing `translations/` with `translations_claude/` for the five pilot questions is a free extra result: it shows whether two different models build similar structures for the same question.
