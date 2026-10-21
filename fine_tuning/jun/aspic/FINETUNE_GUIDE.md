# ASPIC+ fine-tuning guide

This guide covers the full run, from the Gemini translations to the evaluated checkpoints. Every command runs from `fine_tuning/jun/aspic/` with the venv active.

## What this run tests

Llama 3.2 3B Instruct (4-bit, MLX LoRA) is fine-tuned to turn a raw LEET-Arg question into an ASPIC+ translation. The solver in `check_aspic.py`, not the model, then computes each statement and the answer.

The run answers two questions:

1. Does fine-tuning teach the model to write translations the solver can read? This is the **valid** rate.
2. Do those translations lead to the right statements? This is **statement accuracy**, reported separately for structure statements and for weakens/strengthens statements.

If only the first improves, the model learned the format but not the judgment, which is still worth reporting.

## How the model learns

There are two stages, and the gold answer is used only in the first.

1. Building the data (no learning): Gemini writes a translation. The solver turns it into an answer. If the answer matches gold, the translation is kept. If not, Gemini gets feedback and tries again, and after 3 failed tries the question is dropped.
2. Training: the model sees the question and the kept translation, and learns to predict the translation token by token (supervised fine-tuning with a cross-entropy loss). With `--mask-prompt`, only the translation tokens count toward the loss. The solver never runs during training.

Validation loss measures how closely the model copies the teacher's exact tokens. Most of those tokens are format and passage text, so a low loss does not mean correct statements (the Oct 3 run showed this). Checkpoints are therefore chosen by running the solver on each checkpoint's output (`eval_aspic.py`).

## Steps at a glance

| Step | What | Time |
|---|---|---|
| 1 | One-time setup | 10 min |
| 2 | Remove the old low-effort translations | 1 min |
| 3 | Smoke test on your Mac | 30 min |
| 4 | Gemini translations, 73 questions | about 1 to 2 min per question with 3 or 4 tabs |
| 5 | Build the training data | 2 min |
| 6 | Train | about 40 min for 240 iterations |
| 7 | Evaluate the base model and the checkpoints | 1 to 2 hours, unattended |

## 1. One-time setup

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

The scripts find the benchmark file and `../data/split.json` on their own, so you no longer need the `B=` and `S=` variables.

## 2. Remove the old low-effort translations

The 8 files in `translations/` were made with Flash at low effort. All training data now comes from Flash at medium effort, so these are redone in step 4. Delete them so `translations/` starts empty:

```bash
rm translations/*.json
```

They stay in git history (commit `db51dd6`) if anyone asks for them. `translations_claude/` stays as it is. Its five translations are the worked examples in every Gemini prompt, and they are never training data.

## 3. Smoke test

This checks the whole loop on your Mac before you spend days on Gemini: data format, memory, training, and scoring. The numbers mean nothing, because the model is trained and scored on almost the same questions.

```bash
python build_aspic_data.py --smoke --translations translations_claude --out data_smoke

mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train --data ./data_smoke --mask-prompt \
  --iters 20 --batch-size 1 --num-layers 16 --learning-rate 1e-4 \
  --steps-per-eval 10 --save-every 10 \
  --max-seq-length <from the build output> --grad-checkpoint \
  --adapter-path ./adapters_smoke

python eval_aspic.py --adapters adapters_smoke --iters 20 --limit 3 --tag smoke --stats data_smoke/stats.json
```

What to check:

- Training finishes and prints a peak memory number. Note it.
- No line says `Some sequences are longer than ... tokens`. If one does, raise `--max-seq-length`.
- The eval prints one line per question and a table at the end.

Then clean up:

```bash
rm -rf data_smoke adapters_smoke eval_results
```

## 4. Gemini translations

### Settings

- Gemini Flash, medium effort, for every question. Do not change the model or effort halfway, so the whole set comes from one teacher.
- A new chat for every question. Corrections for the same question go into the same chat.
- If you ever change the setting, pass it to the script so the log records it: `python teacher.py --teacher gemini-flash-high save ...`.

### Build the prompts once

```bash
python teacher.py prompts
```

This writes `prompts/<id>.md` for the 58 train and 15 validation questions. The 16 final test questions are never included. Each prompt has the rules, the format, three worked examples from `translations_claude/` and the question with its expert rationale. A question is never shown its own translation as an example.

Validation questions are translated too. Their translations only feed the validation loss during training. All 15 validation questions are evaluated in step 7 whether or not Gemini's translation for them matched.

### The loop, with several Gemini tabs

Each answer takes 2 to 3 minutes, so work on 3 or 4 questions at once, each in its own Gemini tab. Keep one terminal open with:

```bash
python teacher.py run
```

It reacts to two things:

1. **Press Enter** and the next question's prompt is copied to the clipboard. Paste it into a new Gemini tab and send it. Press Enter again for the next tab.
2. **Copy a finished answer** with Gemini's copy button, and the script notices the new clipboard content by itself. It reads the question ID from the `"id"` in the answer, saves the raw answer to `gemini_raw/`, and runs the solver. You don't type anything.

After an answer is checked:

- On MATCH, the translation is saved to `translations/<id>.json`. Close that tab and press Enter for a new question.
- On WRONG, UNRESOLVED or ERROR, the script prints what came out wrong and copies a feedback message that starts with the question ID, for example `[2021_02] Checking your translation ...`. Paste it into that question's tab before you copy anything else.
- After the 3rd failed try, the question is dropped automatically.

After every event the script prints `waiting for:` with the questions that are still open, so you can match them to your tabs. Naming each tab is not needed: the ID is in the prompt and in every feedback message.

Commands you can type in the same terminal:

| Type | What it does |
|---|---|
| Enter | copy the next question's prompt |
| `r ID` | copy ID's prompt again, if you closed its tab by mistake (the try count continues) |
| `a ID` | save the answer on the clipboard as ID, when the script says it could not tell which question an answer is for |
| `d ID reason` | drop a question, for example `d 2021_14 failed on wording the format cannot express` |
| `s` | progress so far |
| `q` | quit. Answers copied after this are not saved until you start `run` again |

Prompts that were handed out are kept in `teacher_sent.json`, so after a restart Enter never gives you a question that is already open in a tab.

If you drop a question by mistake, delete its `DROPPED` line from `teacher_log.csv`.

### One question at a time

The same steps also work as separate commands, for a single tab or for a machine without the macOS clipboard:

```bash
python teacher.py next            # copy the next prompt (or: next 2021_02)
python teacher.py save 2021_02    # check the answer on the clipboard
python teacher.py drop 2021_14 "reason"
python teacher.py status
```

### Reading the saved translations

You don't need to read all 73. Read a translation when `save` prints a warning that a `passage` claim may not be in the text. That warning means Gemini may have invented or paraphrased something the student will learn to copy.

### Without a clipboard

On a machine without `pbcopy`, `run` does not work. Use files instead: `python teacher.py next --file` prints the prompt's path, and `python teacher.py save 2021_02 --file answer.txt` reads the answer from a file.

## 5. Build the training data

```bash
python build_aspic_data.py
```

It writes `data_aspic/train.jsonl` and `data_aspic/valid.jsonl`, one question per line in MLX chat format:

- system: the frozen instruction from `ASPIC_FORMAT.md`
- user: passage, question and statements, without the choice list
- assistant: the translation as compact JSON, with `note`, `summary`, `source` and `id` removed

Before writing, it refuses final test questions, keeps train and validation separate, and re-checks that every translation still reproduces the gold answer.

It prints three things worth noting:

- `--max-seq-length` for training, from the longest example. The pilot examples were under 2,000 tokens.
- How many train and validation questions have no translation.
- The first time, it writes `statement_types.json`: every validation statement tagged as `structure` or `weakens/strengthens` by keyword. Read it once and fix wrong tags by hand; the script never overwrites it. The keyword rule tags 22 of the 51 validation statements as weakens/strengthens.

## 6. Train

```bash
mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train --data ./data_aspic --mask-prompt \
  --iters 240 --batch-size 1 --num-layers 16 --learning-rate 1e-4 \
  --steps-per-eval 20 --val-batches -1 --save-every 20 \
  --max-seq-length <from step 5> --grad-checkpoint --seed 42 \
  --adapter-path ./adapters_aspic
```

| Flag | Why |
|---|---|
| `--mask-prompt` | Loss only on the translation, not the question |
| `--iters 240` | About 4 passes over ~58 questions. If far fewer questions matched, use about 4 times the number of training lines, rounded to a multiple of 20 |
| `--num-layers 16`, `--learning-rate 1e-4`, `--batch-size 1` | Same as the Oct 3 run, so the two runs are comparable |
| `--save-every 20` | A checkpoint every 20 iterations, to choose from in step 7 |
| `--val-batches -1` | Validation loss over all validation lines. Logged for the record; not used to choose |
| `--max-seq-length` | From step 5, so no example is cut |
| `--grad-checkpoint` | Lower memory, a little slower |

Copy the training log (loss per step) into your notes. If training stops with an out-of-memory error, close other apps first, then try `--num-layers 8` and say so when you report.

## 7. Evaluate

Base model first. It gets the same prompt plus one worked example (2022_17), because it has never seen the format:

```bash
python eval_aspic.py --base
```

Then every 40 iterations:

```bash
python eval_aspic.py --adapters adapters_aspic --iters 40 80 120 160 200 240
```

Then the two checkpoints next to the best one, for example `--iters 100 140` if 120 was best.

Each model takes about 10 minutes for the 15 questions. Results are saved after every question, so a stopped run continues where it left off with the same command. To print the table again:

```bash
python eval_aspic.py --summary
```

### Reading the table

| Column | Meaning |
|---|---|
| parsed | the output contains JSON |
| valid | the solver can read it (all IDs defined, every statement has a check) |
| stmt | statements whose computed true/false matches gold. Invalid translations and unsupported statements count as wrong |
| structure | the same, for structure statements |
| weak/str | the same, for weakens/strengthens statements |
| questions | computed answer equals the gold answer |

Choose the checkpoint with the highest `stmt`. On a tie, take the higher `valid`, then the earlier iteration.


The raw output of every question is in `eval_results/<model>.json` under `output`, for failure analysis.

### The final test set

The chosen checkpoint is picked on the validation set, so its validation score is a little optimistic. The honest number comes from running it once on the 16 final test questions. The scripts refuse those questions on purpose. Do that only after the team agrees, with a separate script.

## What to commit

| Commit | Do not commit |
|---|---|
| the scripts and this guide | `.venv/` |
| `prompts/`, `gemini_raw/`, `teacher_log.csv`, `teacher_sent.json` | `adapters_aspic/` (large; already in `.gitignore`) |
| `translations/` | `data_smoke/`, `adapters_smoke/` |
| `data_aspic/`, `statement_types.json` | |
| `eval_results/` | |

A chat model never gives the same answer twice, so the prompts, raw answers and log are the only record of how the training data was made.

## Files

| File | What it does |
|---|---|
| `teacher.py` | The Gemini loop: prompts, the multi-tab `run` mode, check, feedback, retries, log |
| `build_aspic_data.py` | Builds `data_aspic/` from `translations/`, with the safety checks and token lengths |
| `eval_aspic.py` | Runs the base model or checkpoints on the validation questions and scores them with the solver |
| `check_aspic.py` | The solver and the check, unchanged from the pilot |
| `test_base_model.py` | The pilot's base-model test through Ollama. Its prompt helpers are reused by the new scripts |
| `ASPIC_FORMAT.md` | The format and the frozen instruction |
| `make_gemini_batch.py`, `GEMINI_PROMPT.md` | The pilot's batch workflow, replaced by `teacher.py` |
