# ASPIC+ translation and fine-tuning

This folder tests one idea of the project: a small model writes the argument structure of a LEET-Arg question in an ASPIC+ format, and a solver, not the model, computes the answer.

It has two parts:

1. **Pilot (Oct 15, 2026):** can LEET-Arg questions be written in this format so that the solver reproduces the gold answer?
2. **Fine-tuning run (Oct 19 to 20, 2026):** can Llama 3.2 3B Instruct learn to write the format from the raw question? To reproduce it, follow `FINETUNE_GUIDE.md`.

**Result:** fine-tuning improved the format a little and did not improve statement accuracy. Reasoning quality is not measured yet. Details are in [Fine-tuning run](#fine-tuning-run).

## Summary in plain words

### The idea

The small model does not answer the question. It only writes down the argument structure of the passage: who argues what, from which premises, with which rules. A solver then computes whether each statement is true or false. The model does the reading, and the solver does the logic.

### The pipeline and where each number comes from

```
LEET-Arg question (passage + question + statements, without the ① to ⑤ choices)
        │
        ▼
Fine-tuned Llama 3.2 3B
        │
        ▼
ASPIC+ translation (JSON) ────────────►  [1] Format: can the solver read it?
        │
        ▼
Solver (check_aspic.py, PyArg)
        │
        ▼
True/False for each statement ────────►  [2] Statement accuracy: does it match the gold label?
        │
        ▼
Answer ① to ⑤ ────────────────────────►  [3] Question accuracy: does it match the gold answer?

Not measured yet:                         [4] Reasoning quality: is the structure itself a correct reading of the passage?
```

[2] is the same metric as in Trinav’s baseline. The difference is only how the True/False is produced: his model writes it directly, here the solver computes it from the model’s structure.

### What an ASPIC+ translation looks like

One JSON object per question. It lists:

- **positions:** the speakers (for example Alice and Bob) and the main claim of each
- **claims:** every sentence the argument uses, with an ID (`c1`, `c2`, ...)
- **premises and rules:** what each speaker starts from, and which claims follow from which (“if c5 then c6”)
- **contraries and undercuts:** which claims conflict, and which facts block a rule
- **statements:** for each statement (a), (b), (c), a check that the solver can run, for example “B can argue c7, and A cannot”

The translation never says whether a statement is correct. Example from question 2022_17, statement (a) “Unlike A, B argues that evil is a privation”:

```json
{"op": "and", "args": [
  {"op": "holds", "by": "B", "claim": "c7"},
  {"op": "not", "arg": {"op": "holds", "by": "A", "claim": "c7"}}
]}
```

The solver builds B’s and A’s arguments from their premises and rules and checks whether each one can reach c7 (“Evil is a privation”). The full format is in `ASPIC_FORMAT.md`.

### How the training data was made

1. Gemini translated each question into this format. It saw the question and the expert rationale.
2. The solver checked each translation. A translation was kept only if the solver got the gold answer from it. Otherwise Gemini got feedback and tried again.
3. All 73 non-test questions passed (58 for training, 15 for validation). The 16 final test questions were never translated, trained on or evaluated.

At test time the model sees only the question, not the rationale.

### Loss is not accuracy

- **Training and validation loss** measure how closely the model copies Gemini’s JSON, token by token. They say nothing about whether the statements come out right. The solver never runs during training.
- **Accuracy** comes from running the solver on what the model writes, after training.

The two disagree here. Validation loss was lowest at iteration 60 (0.288) and went up to 0.407 at iteration 240. But iteration 240 gave the best solver results. So checkpoints were chosen by running the solver, not by loss.

### Results of the best checkpoint

Run 1, iteration 240, on the 15 validation questions (51 statements). All numbers are after the bracket repair described under [Evaluation](#evaluation).

| Metric | Question it answers | Fine-tuned | Reference |
|---|---|---|---|
| [1] Format | Can the solver read the translation? | 9/15 | Base model with one example: 5/15 |
| [2] Statement accuracy, valid only | Among the 31 statements in readable translations, how many True/False are right? | 19/31 | Calling every statement false: 16/31 |
| [2] Statement accuracy, all | Same, but unreadable translations count as wrong | 19/51 | Calling every statement false: 30/51 |
| [3] Question accuracy | Is the ① to ⑤ answer right? | 3/15 | Always answering ①: 8/15 |
| [4] Reasoning quality | Is the structure a correct reading? | not measured | |

The two versions of [2] follow the reporting rule from the Oct 22 meeting: one counts only parsable outputs, the other counts unparsable outputs as wrong.

**In short:** the format improved a little. When the solver can read the output, it gets about as many statements right as calling every statement false. Reasoning quality is not measured yet.

### Why statement accuracy is not reasoning quality

Two validation questions from the same checkpoint:

- **2022_12: all three statements right, and the answer is right, but the structure is weak.** For (a) and (b), the model adds as a premise the same sentence it then tries to prove, so the check is trivially true. For (c), the rule it wrote proves a different claim than the one being checked, so the check is false. The gold labels are T, T, F, so all three match. The answers are right, the reasoning behind them is not.
- **2022_22: the JSON is valid and the solver runs, but all three statements are wrong.** One rule says “if c1 then c1”, and the check for (c) is “Bob holds c2 and Bob does not hold c2”, which is always false.

So statement accuracy alone can overstate (2022_12) or simply miss (2022_22) what the model understood. This is why [4] needs its own metric. The plan is to base it on the paper Pooyan shared on Oct 19 (“Evaluating Legal Reasoning Traces with Legal Issue Tree Rubrics”, ACL 2026), once the format is reliable.

## Files

| File | What it is |
|---|---|
| `FINETUNE_GUIDE.md` | Step-by-step guide for the fine-tuning run: Gemini translations, data build, training and evaluation |
| `ASPIC_FORMAT.md` | The format: fields, check operations, writing rules, and the frozen instruction used for training and inference |
| `check_aspic.py` | The solver and the check: computes each statement from a translation, maps it to ① to ⑤ and compares with the gold answer. No model involved |
| `teacher.py` | The Gemini loop: one prompt per question, checks each answer, sends feedback, keeps only translations that reproduce the gold answer |
| `build_aspic_data.py` | Builds `data_aspic/` from `translations/`, refuses final test questions and re-checks every translation |
| `eval_aspic.py` | Runs the base model or a checkpoint on the 15 validation questions and scores the output with the solver |
| `rescore_aspic.py` | Rescores saved results with a bracket repair, and adds the judgment columns described below |
| `test_base_model.py` | The pilot's base-model test through Ollama. Its prompt helpers are reused by the scripts above |
| `translations_claude/` | The five pilot translations, written by Claude. Used only as worked examples in the Gemini prompts, never as training data |
| `translations/` | Gemini's 73 translations that reproduce the gold answer (58 train, 15 validation) |
| `prompts/`, `gemini_raw/`, `teacher_log.csv` | The prompt sent for each question, Gemini's raw answers, and one log line per try |
| `data_aspic/` | The MLX training files actually used (58 train, 15 validation) and their token statistics |
| `statement_types.json` | Each validation statement tagged as `structure` or `weakens/strengthens`, checked by hand |
| `train_log.txt`, `train_log_480.txt` | Training logs of the two runs |
| `eval_results/` | Raw outputs and scores for every model evaluated, with `summary.md` and `summary_repaired.md` |
| `model_run1_iter240/` | The fine-tuned LoRA adapter reported below (run 1, iteration 240) |
| `make_gemini_batch.py`, `GEMINI_PROMPT.md`, `gemini_input_*.md`, `gemini_output_*.json` | How the pilot's Gemini rounds were run. Replaced by `teacher.py` |
| `requirements.txt`, `.gitignore` | PyArg version for the venv; keeps `.venv/` and the full checkpoint folders out of git |

## Environment (venv)

The scripts need Python 3.9 or newer. The solver needs PyArg (`python-argumentation`, pinned in `requirements.txt`); without it the built-in solver is used and the results are the same. Training and evaluation also need `mlx-lm` on an Apple Silicon Mac.

```bash
cd fine_tuning/jun/aspic
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install mlx-lm
```

In every new terminal:

```bash
cd fine_tuning/jun/aspic
source .venv/bin/activate
```

## Fine-tuning run

### Data

- **Teacher:** Gemini Flash 3.8 at medium effort, a new chat for every question, through `teacher.py`. Each prompt had the format spec, three worked examples from `translations_claude/`, and the question with its expert rationale. A question was never shown its own translation as an example.
- **Filter:** a translation was kept only if `check_aspic.py` turned it into the gold answer. A failed try got automatic feedback in the same chat. All 58 training and 15 validation questions passed, and none were dropped.
- **Test set:** the 16 final test questions were never translated, trained on or evaluated.
- **Training files:** `data_aspic/`, one question per line in MLX chat format. The input is the frozen instruction plus the passage, question and statements, without the choice list. The target is the translation as compact JSON, without the `note`, `summary`, `source` and `id` fields.
- **Overlap check:** no validation or test question shares a passage with a training question. The closest pair (2023_20 and 2022_16) shares 44% of its longer words but is about different topics.

A note on the log: some questions were re-run after answers were pasted into the wrong tab, and their `DROPPED` lines were removed from `teacher_log.csv` to allow that. This reset the try counter, so for those questions the files in `gemini_raw/` and the log rows do not line up exactly, and the first-try match rate printed by `teacher.py status` is not meaningful. The saved translations are what count: `build_aspic_data.py` re-checks each one against the gold answer.

### Training

`mlx-community/Llama-3.2-3B-Instruct-4bit` with MLX LoRA on 16 layers, learning rate 1e-4, batch size 1, `--mask-prompt` (loss on the translation only), `--grad-checkpoint`, seed 42. The full settings are in `model_run1_iter240/adapter_config.json`.

| Run | Iterations | Checkpoints | Time on a MacBook Pro (M5, 16GB RAM) |
|---|---|---|---|
| Run 1 | 240 (about 4 passes over the training set) | every 20 | about 40 min |
| Run 2 | 480, fresh start, same data and seed | every 40 | about 1.5 h |

The training loss was 0.027 at iteration 240, so the model had nearly memorized the training set by then.

### Evaluation

`eval_aspic.py` runs a model on the 15 validation questions with greedy decoding (up to 2,700 tokens per answer) and passes each output to `check_aspic.py`. The fine-tuned model gets exactly the training prompt. The base model gets the same prompt either alone (0-shot) or with one worked example (1-shot, question 2022_17, a training question), because it has never seen the format.

| Measure | Meaning |
|---|---|
| Valid | The output is JSON that the solver can use: every claim, rule and position it mentions is defined, and every statement has a check |
| Statements | Statements whose computed true/false matches the gold label, out of all 51. Statements of invalid translations count as wrong |
| In valid: model | Statements right, counting only questions with a valid translation (after the bracket repair). This measures statement accuracy without the format failures. It does not measure reasoning quality |
| In valid: “all false” | What calling every statement false gets on exactly the same statements. The fair reference for the column before |
| Questions | Computed answer ① to ⑤ equals the gold answer |

`rescore_aspic.py` repairs brackets mechanically (swaps a wrong closing bracket, adds missing ones) on outputs that ended on their own, and does the same for every model. It added 0 to 2 valid translations per model.

### Results

| Model | Valid | Valid after repair | Statements (before repair) | In valid: model (after repair) | In valid: “all false” (after repair) | Questions (before repair) |
|---|---|---|---|---|---|---|
| Base, 0-shot | 0/15 | 0/15 | 0/51 | | | 0/15 |
| Base, 1-shot | 3/15 | 5/15 | 5/51 | 8/15 | 9/15 | 0/15 |
| Run 1, iteration 240 | 7/15 | 9/15 | 14/51 | 19/31 | 16/31 | 2/15 |
| Run 2, iteration 240 | 4/15 | 5/15 | 8/51 | 10/15 | 7/15 | 1/15 |
| Run 2, iteration 320 | 6/15 | 6/15 | 11/51 | 11/20 | 8/20 | 0/15 |
| Run 2, iteration 400 | 6/15 | 7/15 | 10/51 | 11/21 | 12/21 | 0/15 |
| Run 2, iteration 480 | 2/15 | 3/15 | 3/51 | 5/9 | 6/9 | 0/15 |

For reference, on these 15 questions, always answering ① gets 8/15 questions, and calling every statement false gets 30/51 statements. Every checkpoint is listed in `eval_results/summary.md` and `eval_results/summary_repaired.md`.

### What the results show

1. **Format improved a little, statement accuracy did not** (see the [summary](#summary-in-plain-words)). This holds for every checkpoint: among valid translations, the model is between 3 statements above and 3 below calling every statement false, and the base model is 1 below. With 9 to 31 statements per row, a difference of 3 is within chance.
2. **Training longer made it worse.** At 480 iterations only 2 of 15 translations were valid. The late errors look like memorization: rules that use claim IDs never defined for this question, positions that are not in the passage (“Jurors”, “Experiment”, “American researchers”), and an invented check operation (“mitigates”).
3. **The same setup gave different results.** Run 2 repeated run 1's data, settings and seed, but at iteration 240 it wrote 4 valid translations instead of 7. The two training logs are identical through iteration 40 and then diverge (128,687 vs 129,297 trained tokens at iteration 120). In mlx-lm's trainer, batches are shuffled with numpy's global random state, and every validation pass draws from it too, so validating every 20 vs every 40 iterations changed the order of the second pass over the data. Re-evaluating run 1's checkpoint 240 gave identical numbers, so the evaluation is deterministic and the difference comes from training order alone.
4. **The failure types change over training.** Early checkpoints get stuck repeating claims or statements until the token limit. Middle ones lose count of brackets inside deeply nested `add` blocks. Late ones make the memorization errors in point 2.

### Limitations

- 15 validation questions is a small sample, and the checkpoint was chosen on the same questions, so its score is slightly optimistic.
- The final test set has not been run.
- Reasoning quality is not measured. All accuracy numbers here are True/False per statement or ① to ⑤ per question.
- The training data comes from one teacher model and contains only translations that reproduce the gold answer.
- ASPIC+ has no support relation, so “strengthens” statements are expressed indirectly (see pilot observation 2).
- The model was trained and evaluated on LEET-Arg only.

### Using the adapter

`model_run1_iter240/` holds run 1's checkpoint at iteration 240, the best one on validation:

```python
from mlx_lm import load, generate
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit", adapter_path="model_run1_iter240")
```

The full checkpoint folders are not in the repo (about 345 MB per run). `FINETUNE_GUIDE.md` rebuilds them.

### Next steps

1. **Test the solver with good extraction.** Have Gemini translate the 15 validation questions without the expert rationale, in one try with no feedback, and run the solver. If the solver cannot reach good answers from a strong model's structures, training a small model to write them will not help.
2. **If the small model stays in scope,** fix the format without training (constrained decoding), so that only the reasoning is measured, or try a 7 to 8B local model with one example.
3. **If a claim depends on one checkpoint,** repeat the run with two or three different example orders and report the range.

## Pilot (Oct 15, 2026)

Run the check on the pilot translations:

```bash
python check_aspic.py \
    --bench ../../../benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
    --split ../data/split.json -v translations_claude/
```

The script refuses the 16 final test questions and warns about validation questions. When PyArg is installed, it computes the grounded extension with PyArg and checks that the built-in solver agrees.

| Question | Category | Statement types | Statements | Answer | Gold |
|---|---|---|---|---|---|
| 2021_37 | Argument Evaluation & Problem Solving | strengthens, weakens | F T F | ② | ② |
| 2022_17 | Argumentation and Rebuttal | holds, assumes | T T T | ⑤ | ⑤ |
| 2022_29 | Argument Evaluation & Problem Solving | strengthens, weakens | T F T | ③ | ③ |
| 2023_26 | Argument Analysis | weakens, assumes, holds | T F F | ① | ① |
| 2024_15 | Argument Evaluation & Analysis | status under a criterion, holds | T T T | ⑤ | ⑤ |

All 5 reproduce the gold answer, and no statement had to be marked unsupported. To make sure the checks are not trivially true, each translation was also broken on purpose (an undercut or a premise removed, a criterion made defeasible, and so on). Each time the expected statement flipped and the answer became wrong.

### Observations

1. **40 of 102 claims and rules come from the expert rationale, not the passage.** At test time the model only sees the passage, so it has to supply these itself. The share is highest for Argument Analysis (2023_26: 12 of 18), where the statements are about assumptions the passage never states.
2. **For weakens/strengthens (7 of 15 statements), the translation decides the answer.** Whether the solver says "weakened" depends on whether the translation includes the rule that connects the evidence to the claim. The solver does real work only where there is a counter-argument: in 2022_29 (b) and 2021_37 (c), the tempting attack is defeated by an undercut (the 20-year lag; caloric is massless), and the solver computes that the claim survives.
3. **One format change was needed:** named hypotheticals, because 2024_15 reuses the same "if criterion K" block in two statements.
4. **Interpretation call:** 2023_26 (a) "method A is difficult to apply" is translated as "A's assumption is undermined, so A is weakened".

The five pilot translations were written by Claude from the public LEET-Arg text. The pilot's Gemini rounds are described in `GEMINI_PROMPT.md`; for the full set they were replaced by `teacher.py`.