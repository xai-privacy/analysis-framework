# Fine-tuning Llama 3.2 3B Instruct with expert rationales

First fine-tuning round for [issue #13](https://github.com/xai-privacy/analysis-framework/issues/13). The model is trained to give a CORRECT/INCORRECT verdict for each statement, with the expert rationale from the benchmark as its reasoning. The setup follows the [fine-tuning README](../README.md).

**Result:** fine-tuning taught the model the answer format, but it did not improve accuracy over the base model on the 15 validation questions.

## Files

Everything for this run is in `fine_tuning/jun/`, so it doesn’t overwrite other team members’ data or adapters. The shared `fine_tuning/data/train.jsonl` and `valid.jsonl` are left empty. The scripts find the repo root on their own, so they run from any folder.

| File | What it does |
|---|---|
| `leet_arg_format.py` | Shared helpers: builds the prompt for each question, derives the per-statement gold labels, builds the training answers, and reads verdicts out of model answers |
| `prepare_leet_arg.py` | Builds `data/train.jsonl`, `data/valid.jsonl` and `data/split.json` from the benchmark, with the final test questions removed |
| `evaluate_leet_arg.py` | Runs the base or a fine-tuned model on `data/valid.jsonl` and saves every answer with its scores |
| `rescore_results.py` | Re-scores saved results without running the model again, with a strict and a lenient parser |
| `data/train.jsonl`, `data/valid.jsonl` | 58 training and 15 validation questions in MLX chat format |
| `data/split.json` | The question IDs in each split, and the reason each excluded question was left out |

`fine_tuning/data/train_arg_framework_structure.jsonl` and the committed `fine_tuning/adapters/` folder are not used here. Adapters trained in this folder are kept out of the repo by `.gitignore`.

## Data

**Source:** `benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json` (93 questions).

**Excluded (20 questions):**

- the 16 final test questions in `testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json`, removed before anything else
- 2023_19, 2024_25 and 2025_21: diagram questions with no per-statement rationale to train on
- 2025_16: the choice list says ② = (b), but the rationale says only (c) is correct and that the answer is ② ([#14](https://github.com/xai-privacy/analysis-framework/issues/14))

**Split:** the remaining 73 questions are split 80/20 by question, stratified by category, with seed 42. That gives 58 training and 15 validation questions. `prepare_leet_arg.py` checks that no final test question ends up in either file.

**Input:** the passage, the question and the statements. The model never sees the ① to ⑤ options.

**Labels:** one CORRECT/INCORRECT label per statement. The dataset has no per-statement label field, so the labels are derived from the gold answer, the choice list and the question wording. For example, in 2021_02 the answer is ② and ② is “(b)”, so (b) is CORRECT and (a) and (c) are INCORRECT. For single-answer questions, each option counts as one statement. For “NOT” questions (“Which of the following is NOT a correct analysis?”), the chosen option is the INCORRECT one; this matches how `train_arg_framework_structure.jsonl` labels them. Every derived label set maps back to the gold answer, and the labels agree with the wording of the expert rationales except for 2025_16.

**Training answer:** a summary of the debate from the rationale’s preliminaries, then reasoning and a verdict for each statement:

```
Summary: Alice, Bob, and Charlie are debating whether obscene materials ...

Statement (a)
Reasoning: (a) Alice’s position is to exclude moral standards such as obscenity ...
Verdict: INCORRECT

Statement (b)
...
```

Sentences that name the final choice (“so the correct answer is ②”) are removed from the rationales, since the model never sees the choices.

The reasoning is the expert rationale text, not the Premise/Inference Rule structure in the fine-tuning README example, because the translation of the benchmark into that structure is still in progress.

**Format:** each line uses MLX’s chat format (`{"messages": [system, user, assistant]}`), not hand-written `"text"` with Llama tags. With `"text"`, MLX adds a second `<|begin_of_text|>` token, and `mlx_lm.generate` applies the chat template again on top of the hand-written one, so testing would not match training. With the chat format, MLX applies the Llama 3.2 template the same way in both.

## How to reproduce

From `fine_tuning/jun/`, with the virtual environment from the [fine-tuning README](../README.md) active:

```bash
pip install mlx-lm
python3 prepare_leet_arg.py

mlx_lm.lora \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --train --data ./data --mask-prompt \
  --iters 180 --batch-size 1 --num-layers 16 --learning-rate 1e-4 \
  --steps-per-eval 20 --save-every 20 \
  --adapter-path ./adapters_leet

python3 evaluate_leet_arg.py --out results/base.json
python3 evaluate_leet_arg.py --adapter-path ./adapters_leet --out results/finetuned.json
python3 rescore_results.py results/base.json results/finetuned.json
```

`--mask-prompt` computes the loss only on the answer, not on the passage ([mlx-lm docs](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)). Without it, about half of each training example would be the passage text. Training starts from the base model; it does not load an existing adapter.

To evaluate a saved checkpoint, copy it into its own folder:

```bash
mkdir -p adapters_leet_0000020
cp adapters_leet/adapter_config.json adapters_leet_0000020/
cp adapters_leet/0000020_adapters.safetensors adapters_leet_0000020/adapters.safetensors
python3 evaluate_leet_arg.py --adapter-path ./adapters_leet_0000020 --out results/finetuned_0000020.json
```

`evaluate_leet_arg.py` saves after every question. If it is stopped, the same command continues where it left off.

## Results

Training ran on a MacBook Pro at about 0.2 to 0.27 iterations per second, with peak memory of 10.2 GB.

| Iteration | 1 | 20 | 40 | 60 | 80 | 100 | 120 | 140 | 160 | 180 |
|---|---|---|---|---|---|---|---|---|---|---|
| Validation loss | 1.423 | **1.265** | 1.284 | 1.287 | 1.421 | 1.369 | 1.531 | 1.667 | 1.748 | 1.727 |

Training loss went from about 1.25 to 0.25. It dropped sharply between iterations 60 and 70, right after the second pass over the 58 training questions began, so from there the model was memorizing the training answers. This is why I did not go up to the fine-tuning README’s 600 iterations.

Results on the 15 validation questions (51 statements):

| Model | Parse rate | Statements | Questions |
|---|---|---|---|
| Base | 0.93 | 27/51 | 3/15 |
| Fine-tuned, iteration 20 | 0.67 | 15/51 | 1/15 |
| Fine-tuned, iteration 60 | 0.53 | 12/51 | 0/15 |
| Fine-tuned, iteration 180 | 1.00 | 26/51 | 3/15 |

- **Parse rate** is the share of questions where every statement got a verdict.
- **Statements** counts per-statement verdicts that match the gold label. A missing verdict counts as wrong.
- **Questions** maps the verdicts back to ① to ⑤ and compares with the gold answer.

For reference, always answering INCORRECT gets 30/51 statements, and always answering ① gets 8/15 questions on this split.

**What the results show:**

- **Only the format improved.** At iteration 180 the model always answers in the training format (parse rate 1.00), but its accuracy is the same as the base model’s. Both are around coin-flip on statements, and each got a different 3 questions right.
- **The fine-tuned model says CORRECT too often:** 36 of 51 verdicts, while the gold labels have 21. It often restates the statement and concludes that it is correct.
- **The earlier checkpoints are worse because they often give no verdict at all.** At iteration 60, 6 of the 7 answers with missing verdicts got stuck repeating text until they hit the token limit.
- **Lower validation loss did not mean better verdicts.** The loss covers the whole answer, about 540 tokens on average, and the verdicts are only a few of those tokens. So it measures how closely the model copies the expert wording, not whether its verdicts are right.

**Scoring the base model:** the base model often writes verdicts in its own format, for example “(a) **INCORRECT**” or “Correctness: CORRECT”. The strict parser in `evaluate_leet_arg.py` only reads the training format and scores the base model at 0/15. The base row above uses the lenient parser in `rescore_results.py`, which accepts any clear upper-case CORRECT/INCORRECT. The lenient scores match a manual read of all 15 base outputs. The fine-tuned results are the same with either parser.

## Limitations

- **15 validation questions is a small sample.** A difference of one or two questions is noise.
- **The same 15 questions were used to look at checkpoints and to report results.** The clean comparison will come from the final test set.
- **“NOT” questions are a weak spot.** The question asks for the option that is NOT correct, but the training answer gives a verdict on every option. There are 3 such questions in training and 1 in validation (2024_05).
- **The reported adapters were trained on a slightly earlier `train.jsonl`.** In that version, record 2023_25 still had a stray “<Choices> Analysis” tag in its summary. The current file has it removed; it is the only difference.

## Next steps

- When the translation of the benchmark into the Premise/Inference Rule structure is ready, retrain on it with the same split and the 16 final test questions removed. That shows whether a structured answer helps where the rationale text did not.
- Fine-tuning may still be useful for format. Frontier APIs get format compliance from constrained decoding (structured outputs). On a local model, fine-tuning could do a similar job for the extraction format the solver needs, even if it does not improve the reasoning.
