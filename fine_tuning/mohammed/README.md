# LEET-Arg Fine-Tuning — Mohammed

This folder contains my fine-tuning experiments for **Llama 3.2 3B Instruct** on the LEET-Arg benchmark using **MLX-LM LoRA**.

The main goal is to test whether fine-tuning improves reasoning, and to check whether the model learns shortcuts such as always choosing the same answer number.

## Current result

The strongest result so far comes from **removing the ①–⑤ answer choices completely** and training the model to classify each statement as `CORRECT` or `INCORRECT`.

On the same 15 validation questions:

| Model | Statement accuracy | Full-question accuracy | Parse rate |
|---|---:|---:|---:|
| Base model | 17/49 = 34.7% | 1/15 = 6.7% | 73.3% |
| Statement LoRA, iter 20 | 23/49 = 46.9% | 2/15 = 13.3% | 86.7% |
| Statement LoRA, iter 40 | 24/49 = 49.0% | 0/15 | 100% |
| Statement LoRA, iter 60 | 21/49 = 42.9% | 0/15 | 86.7% |
| Statement LoRA, iter 80 | 24/49 = 49.0% | 0/15 | 100% |
| **Statement LoRA, iter 100** | **29/49 = 59.2%** | **4/15 = 26.7%** | **100%** |

The validation labels are almost balanced:

```text
CORRECT:   25
INCORRECT: 24
```

So simple one-label baselines are:

```text
always CORRECT   = 25/49 = 51.0%
always INCORRECT = 24/49 = 49.0%
```

The iteration-100 statement model is above both baselines and predicts both labels:

```text
CORRECT predictions:   19
INCORRECT predictions: 30
```

This result was reproduced by rerunning the full experiment.

## Data

Current benchmark:

```text
94 total questions
```

The 16 final test questions are removed before training or validation.

Current split:

```text
63 training questions
15 validation questions
16 held-out test questions
```

The split is stored in:

```text
data/split.json
data/valid_ids.json
data/test_ids.json
```

The dataset checks confirm:

```text
statement_only: PASS train=63 valid=15
statement_renamed: PASS train=63 valid=15
compact_structured: PASS train=63 valid=15
No held-out test IDs were written.
```

The held-out test set is currently frozen. New experiments use only the 15 validation questions.

## MLX format

Training data uses MLX chat format:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

MLX applies the Llama chat template automatically. The training commands use:

```text
--mask-prompt
```

so loss is calculated on the assistant answer rather than on the full prompt.

## Experiments

### 1. Direct-answer fine-tuning

Target:

```text
Answer: 2
```

This learned the requested output format, but it did not learn stable question-specific answers.

Checkpoint behavior on the validation set:

| Iteration | Main prediction behavior | Accuracy |
|---:|---|---:|
| 20 | 13 × option 1, 2 unparsed | 3/15 |
| 40 | 3 × option 3, 12 unparsed | 1/15 |
| 60 | 15 × option 2 | 5/15 |
| 80 | 11 × option 5, 4 × option 3 | 2/15 |
| 100 | 15 × option 5 | 2/15 |

The iteration-60 accuracy looks higher, but the model simply predicted option 2 for every question. Five validation questions happened to have answer 2.

The final checkpoint predicted option 5 for all 15 questions.

This is evidence of **answer-label shortcut / collapse**, not stable reasoning.

### 2. Rationale + answer fine-tuning

Target:

```text
<expert rationale>

Answer: 2
```

Earlier validation result:

```text
accuracy: 1/15 = 6.7%
parse rate: 80%
```

This model produced more varied answers than the direct-answer model, but it did not improve validation accuracy.

### 3. Choice-number permutation

To test whether the original answer positions were causing the shortcut, the ①–⑤ choices were randomly reordered per question while keeping the correct logical choice unchanged.

Result:

```text
accuracy: 3/15 = 20%
parse rate: 100%
prediction counts: option 5 = 15/15
```

Changing the answer positions did **not** remove the collapse. The model still predicted option 5 for every validation question.

### 4. Statement-only fine-tuning

The answer choices are removed completely.

The model only predicts:

```text
Statement 1: CORRECT
Statement 2: INCORRECT
Statement 3: CORRECT
```

This avoids asking the model to learn the arbitrary mapping from a correct statement set to ①–⑤.

At iteration 100:

```text
statement accuracy: 29/49 = 59.2%
full-question accuracy: 4/15 = 26.7%
parse rate: 100%

predicted CORRECT: 19
predicted INCORRECT: 30
```

Compared with the base model at the statement level:

```text
wrong -> right: 18
right -> wrong: 6
right -> right: 11
wrong -> wrong: 14
```

This is the most promising result so far.

Earlier statement checkpoints still showed collapse:

```text
iter 20: almost all CORRECT
iter 40: all INCORRECT
iter 60: almost all INCORRECT
iter 80: all INCORRECT
iter 100: mixed CORRECT / INCORRECT
```

So checkpoint-level behavior is important and validation loss alone is not enough.

### 5. Renamed-speaker experiment

A version was created that changes repeated names such as Alice/Bob/Charlie.

The first run produced malformed generations and was not reliable enough to interpret. Therefore I am **not** claiming that renaming speakers helped or hurt reasoning.

This experiment needs to be fixed and rerun before drawing a conclusion.

### 6. Compact structured experiment

A compact structured target was also tested.

The first run produced malformed/repetitive outputs, so this experiment is also treated as an unsuccessful implementation rather than evidence against structured reasoning.

A fuller argumentation-framework representation is still planned:

```text
Premise 1
Premise 2
Rule
Application of Rule
Conclusion
```

The structured reasoning content should be manually reviewed or reliably derived before using it for training.

## Reproducing the current experiments

The full validation rerun is in:

```text
rerun_today_experiments.sh
```

Run from `fine_tuning/mohammed/`:

```bash
chmod +x rerun_today_experiments.sh
./rerun_today_experiments.sh
```

The script:

```text
1. prepares the statement datasets
2. validates the split and checks test leakage
3. evaluates the base statement model
4. trains statement-only LoRA
5. evaluates checkpoints 20/40/60/80/100
6. compares statement checkpoints
7. creates the choice-permutation dataset
8. trains and evaluates the choice-permutation model
9. re-evaluates the original direct-answer checkpoints
10. checks for empty result files
```

The normal rerun does **not** evaluate `data/test_ids.json`.

The renamed and compact experiments are optional because their current versions are not yet considered reliable:

```bash
RUN_RENAMED=1 RUN_COMPACT=1 ./rerun_today_experiments.sh
```

## Main files

Important scripts:

```text
audit_dataset.py
prepare_data.py
validate_data.py
evaluate_v2.py
reparse_results.py
compare_results.py

leet_statement_utils.py
prepare_statement_data.py
validate_statement_data.py
evaluate_statements.py
compare_statement_results.py

prepare_choice_permutation.py
evaluate_choice_permutation.py
make_checkpoint_adapter.py
prepare_argument_framework_data.py

run_training.sh
rerun_today_experiments.sh
```

Important data:

```text
data/split.json
data/valid_ids.json
data/test_ids.json

data/direct/
data/rationale/
data/statement_only/
data/statement_renamed/
data/compact_structured/
data/choice_permuted/
```

Important results:

```text
results/base_valid.json
results/base_valid_reparsed.json
results/direct_valid.json
results/direct_valid_reparsed.json
results/rationale_valid.json
results/rationale_valid_reparsed.json

results/direct_20_valid.json
results/direct_40_valid.json
results/direct_60_valid.json
results/direct_80_valid.json
results/direct_100_valid.json

results/base_statement_valid_v2.json
results/statement_valid_v2.json
results/statement_20_valid.json
results/statement_40_valid.json
results/statement_60_valid.json
results/statement_80_valid.json
results/statement_100_valid.json
results/statement_checkpoint_summary.txt

results/choiceperm_valid.json
```

Large adapter files are not committed.

## Main finding

The current experiments suggest:

> Answer-number supervision encourages shortcut behavior. Randomly changing the answer positions did not remove the problem. Removing the answer numbers completely and training the model to make statement-level CORRECT/INCORRECT judgments produced the best validation result so far.

This is still a small validation set, so the next step is to check whether the statement-only result is stable across additional training runs before making a stronger conclusion.

## Next steps

- Repeat the statement-only experiment with additional training seeds while keeping the same train/validation split.
- Analyze which types of statements changed from wrong to right.
- Fix the speaker-renaming experiment and rerun it separately.
- Continue developing the Premise → Rule → Application → Conclusion representation.
- Keep the final test set frozen during method development.
