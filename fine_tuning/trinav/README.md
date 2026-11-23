# Trinav Fine-Tuning Data

This folder prepares LEET-Arg fine-tuning data in the per-statement rationale
format used by Jun's experiment. Each assistant target contains the rationale
summary, reasoning, and a `CORRECT` or `INCORRECT` verdict for every statement.
The answer-choice list is removed from the model input.

The source is
`benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json`.
The 16 final-test questions, three diagram questions without per-statement
rationales, and inconsistent question `2025_16` are excluded before splitting.

The current source has 94 questions, while Jun's run used the earlier 93-question
snapshot. The only added ID is `2021_13`; it has complete rationales now, but is
excluded here to reproduce Jun's cohort and exact split. The split therefore
contains the same 58 training and 15 validation question IDs as Jun's seed-42
split. The extra snapshot exclusion is recorded in `data/split.json`.

Regenerate the data from the repository root or this directory:

```bash
python3 fine_tuning/trinav/prepare_data.py
```

The script writes `data/train.jsonl`, `data/valid.jsonl`, and `data/split.json`.
The ID-only validation and test manifests are also written under `data/`; no
test examples are included in the training or validation JSONL. Keep the test
set untouched when comparing checkpoints and select checkpoints using
validation results.

## Chameleon LEET-Arg Workflow

Run this workflow on your Chameleon Tesla P100 (compute capability 6.0), not
in the local WSL environment. It uses NF4 4-bit base-model weights and LoRA
adapters. The commands below use the original `Qwen/Qwen3-4B` weights, quantized to NF4
at load time. This replaces Jun's earlier MLX-format
`mlx-community/Llama-3.2-3B-Instruct-4bit` checkpoint, which cannot be loaded
by this Transformers/bitsandbytes CUDA workflow.

Authenticate with Hugging Face before starting if the environment or model
requires it. The model is downloaded by Transformers when a run starts; no
model weights need to be copied to the VM.

### Set up the Chameleon VM

From `analysis-framework/fine_tuning/chameleon` on your local machine, upload
the Trinav folder. Replace the key and IP with your own:

```bash
ssh -i fine_tuning/karimm_lease1.pem cc@129.114.108.3 'mkdir -p ~/xai'
scp -i fine_tuning/karimm_lease1.pem -r fine_tuning/trinav/ cc@129.114.108.3:~/xai/
```

Log in, create the environment, and install the pinned dependencies:

```bash
ssh -i fine_tuning/karimm_lease1.pem cc@129.114.108.3
cd ~/xai/trinav
nvidia-smi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-chameleon.txt
python - <<'PY'
import torch
print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("Supported architectures:", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.get_device_capability(0) == (6, 0), "Expected a Tesla P100"
assert "sm_60" in torch.cuda.get_arch_list(), "PyTorch build lacks P100 support"
PY
hf auth login
```

The baseline runners need both source benchmarks on the VM. The LEET-Arg
adapter test also needs its final-test ID list. From the repository root on
your local machine, copy them once (DO NOT DO THIS IF THE FILES EXIST IN THE VM!!!):

```bash
ssh -i fine_tuning/karimm_lease1.pem cc@129.114.108.3 \
	'mkdir -p ~/xai/trinav/benchmarks'
scp -i fine_tuning/karimm_lease1.pem \
	benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	cc@129.114.108.3:~/xai/trinav/benchmarks/
scp -i fine_tuning/karimm_lease1.pem \
	benchmarks/MBE-exam-questions_distilled_dataset_cleaned.json \
	cc@129.114.108.3:~/xai/trinav/benchmarks/
scp -i fine_tuning/karimm_lease1.pem \
	benchmarks/LEET_Arg_Baseline.jsonl benchmarks/MBE-exam-questions-baseline.jsonl \
	cc@129.114.108.3:~/xai/trinav/benchmarks/
scp -i fine_tuning/karimm_lease1.pem \
	testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json \
	cc@129.114.108.3:~/xai/trinav/data/
```

### 1. Baseline and CoT testing

Baseline inputs use the same JSONL format for both datasets and are stored in
`benchmarks/`: each line contains only an ID and question text. LEET-Arg
questions include the situation, question, and statements, but not the
`<choices>` section. MBE questions omit the answer options. Gold answers and
rationales remain in the separate source benchmarks used for scoring and are
not passed to the model.

Run the base model over the complete LEET-Arg source dataset:

You can optionally provide a global system prompt from a UTF-8 text file with
`--prompt path/to/prompt.txt`. The prompt is sent before each question. If
both `--prompt` and `--prompt-type` are omitted, no system prompt is sent. The
separate `--format-hint` option can still add its output-format instructions.
For per-statement verdicts that the evaluator can parse, use the provided
baseline prompt type and `--format-hint` together. Use `--prompt-type cot` to
select the structured reasoning prompt instead:

```bash
python run_hf_questions.py \
	--input benchmarks/LEET_Arg_Baseline.jsonl \
	--gold-data benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	--split data/split.json \
	--all-data \
	--prompt-type baseline \
	--format-hint \
	--model Qwen/Qwen3-4B \
	--run-name qwen3-base \
	--output-root testing/baseline/huggingface
```

For a CoT run, replace `--prompt-type baseline` with `--prompt-type cot` and change the run name to qwen3-cot to put the results in a separate directory.
The matching system prompt is loaded automatically. The available types are
`baseline` and `cot`; use `--prompt path/to/file.md` instead if you need a
custom prompt. `--prompt` and `--prompt-type` cannot be used together.

The evaluator sends only the `question` field from the JSONL file, plus a
system prompt selected by `--prompt-type` or supplied with `--prompt`. The
`--all-data` option is for
base-model baseline runs only. It requires the complete question file and uses
the separate `--gold-data` file only for scoring. Omit it for final-test adapter
evaluation. Do not use final-test results to choose training steps or adapters;
use validation results for those decisions.

### 2. Fine-tuning

Train for 60 steps, evaluating and saving every 15 steps. Validation loss is
used during training to select the best checkpoint. The final adapter is saved
to `adapters_chameleon/`.

```bash
python train_chameleon.py \
	--model Qwen/Qwen3-4B \
	--train-data data/train.jsonl \
	--valid-data data/valid.jsonl \
	--split data/split.json \
	--output-dir adapters_chameleon \
	--max-steps 60 \
	--checkpoint-steps 15 \
	--learning-rate 1e-4 \
	--max-seq-length 4096
```

The trainer uses batch size 1, LoRA rank 8, and a constant learning rate. If an
example exceeds the sequence length, it stops with that example's ID rather
than truncating the rationale. After training completes, `trainer_state.json`
is saved under the output directory. To plot the metrics for the command above
on Chameleon:

```bash
python plot_training_metrics.py \
	--trainer-state adapters_chameleon/trainer_state.json
```

For Phi-4-mini, substitute `microsoft/Phi-4-mini-instruct` for the model ID and
use `adapters_phi4` as the output directory. The trainer detects Phi-4's fused
projection layers automatically.

### 3. Upload the adapter to Hugging Face

The training output is a PEFT adapter, not a standalone model. Upload its
configuration and weights; the base model is loaded separately at inference.
Keep the Hugging Face repo private unless you intend to publish it. From the
VM, replace `YOUR_HF_USERNAME` with your account or organization:

```bash
hf auth login
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi

adapter_dir = Path("adapters_chameleon")
repo_id = "YOUR_HF_USERNAME/leetarg-qwen3-lora"
config = adapter_dir / "adapter_config.json"
weights = next((adapter_dir / name for name in (
    "adapter_model.safetensors", "adapter_model.bin"
) if (adapter_dir / name).is_file()), None)

if not config.is_file() or weights is None:
    raise SystemExit(f"Adapter config/weights not found in {adapter_dir}")

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
for path in (config, weights):
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="model",
    )
print(f"Uploaded adapter to https://huggingface.co/{repo_id}")
PY
```

Preserve the adapter locally and copy it off the VM before the lease ends.

### 4. Test using the uploaded adapter

Run the same test IDs with the same base model and prompt format. You need
Hugging Face access to both the gated base model and the adapter repository.
Replace the adapter placeholder with the repo ID used during upload:

```bash
python run_hf_questions.py \
	--input data/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	--test-ids data/LEET_Arg_Questions_Test_Set.json \
	--split data/split.json \
	--model Qwen/Qwen3-4B \
	--adapter-path YOUR_HF_USERNAME/leetarg-qwen3-lora \
	--run-name qwen3-adapter-hf \
	--output-root testing/baseline/huggingface
```

Each run is saved under
`testing/baseline/huggingface/<run-name>/`, with one Markdown file per question,
`results.json` metrics, and `run.json` settings. Compare the baseline and
adapter run summaries and question files.

#### Troubleshooting CUDA errors

The P100 requires a PyTorch build that includes `sm_60`. The requirements file
pins `torch==2.14.1+cu126`; after installation, confirm the architecture check
in the setup step passes. If a previous environment has a different Torch
version, rerun the `python -m pip install -r requirements-chameleon.txt`
command from setup. If PyTorch reports `sm_60` support but bitsandbytes still
fails, capture the error along with the GPU name, `torch.__version__`,
`torch.version.cuda`, and bitsandbytes version.

#### Troubleshooting chat-template errors

If `tokenizer.apply_chat_template` reports that Jinja2 must be at least 3.1.6,
upgrade it in the active environment:

```bash
python -m pip install -U "jinja2>=3.1.6,<4"
```

Jinja2 is also pinned in `requirements-chameleon.txt` for new installations.

## MBE Dataset

The MBE split is kept separate under `fine_tuning/trinav/mbe/data/`. It uses
`benchmarks/MBE-exam-questions_distilled_dataset_cleaned.json` (1,280
questions), reserves 30 questions for testing, and splits the remaining 1,250
at the same approximate 58:15 train/validation ratio as LEET-Arg. With seed 42,
the result is 993 training and 257 validation questions. Allocation is
stratified by legal domain.

Each train/validation line is a chat example: the user message contains the
question and four choices, and the assistant target contains the answer letter
and the source IRAC explanation. Test lines contain only system/user messages;
their gold answer and explanation are stored separately in `evaluation` and
never appear in the model-facing messages.

Regenerate the split from the repository root (SKIP IF THIS EXISTS!):

```bash
python fine_tuning/trinav/mbe/prepare_mbe_data.py
```

The script writes `train.jsonl`, `valid.jsonl`, `test.jsonl`, `split.json`, and
ID lists under `fine_tuning/trinav/mbe/data/`. The same trainer can fine-tune on
the MBE chat data by pointing it at the MBE train, validation, and split files:

```bash
python fine_tuning/trinav/train_chameleon.py \
	--model Qwen/Qwen3-4B \
	--train-data fine_tuning/trinav/mbe/data/train.jsonl \
	--valid-data fine_tuning/trinav/mbe/data/valid.jsonl \
	--split fine_tuning/trinav/mbe/data/split.json \
	--output-dir fine_tuning/trinav/mbe/adapters_qwen3 \
	--max-steps 180 \
	--checkpoint-steps 20 \
	--learning-rate 1e-4 \
	--max-seq-length 4096
```

To train on Chameleon, copy the `mbe/` folder to the VM as well as the shared
trainer scripts. Use paths relative to `~/xai/trinav`, for example `mbe/data/`
and `mbe/adapters_qwen3/`. Keep `test.jsonl` out of training and validation.

### Run the MBE workflow on Chameleon

The commands below assume the Chameleon VM, `.venv`, and Hugging Face access
were set up as described above. From the repository root on your local machine,
copy the MBE data, evaluator, shared trainer files, and source benchmark to the
VM. Replace the key and IP with your own (SKIP IF DONE DURING SETUP):

```bash
ssh -i fine_tuning/karimm_lease1.pem cc@129.114.108.3 \
	'mkdir -p ~/xai/trinav/benchmarks'
scp -i fine_tuning/karimm_lease1.pem -r fine_tuning/trinav/mbe \
	cc@129.114.108.3:~/xai/trinav/
scp -i fine_tuning/karimm_lease1.pem \
	fine_tuning/trinav/run_hf_mbe.py \
	fine_tuning/trinav/prompt_selector.py \
	fine_tuning/trinav/mbe-baseline-prompt.md \
	fine_tuning/trinav/mbe-cot-prompt.md \
	fine_tuning/trinav/chameleon_utils.py \
	fine_tuning/trinav/train_chameleon.py \
	cc@129.114.108.3:~/xai/trinav/
scp -i fine_tuning/karimm_lease1.pem \
	benchmarks/MBE-exam-questions_distilled_dataset_cleaned.json \
	benchmarks/MBE-exam-questions-baseline.jsonl \
	cc@129.114.108.3:~/xai/trinav/benchmarks/
```

Log in to the VM, activate the environment, and prepare the split:

```bash
ssh -i fine_tuning/karimm_lease1.pem cc@129.114.108.3
cd ~/xai/trinav
source .venv/bin/activate
python mbe/prepare_mbe_data.py \
	--source benchmarks/MBE-exam-questions_distilled_dataset_cleaned.json \
	--out mbe/data
```

Run the base model on the full MBE dataset before fine-tuning. The baseline
input at `benchmarks/MBE-exam-questions-baseline.jsonl` contains only IDs and question text,
with no system prompt, answer choices, answers, or rationales. The source
benchmark is read separately for scoring. Supply your own system prompt with
`--prompt` or select one with `--prompt-type`; no system prompt is added by
default if neither option is supplied.

```bash
python run_hf_mbe.py \
	--input benchmarks/MBE-exam-questions-baseline.jsonl \
	--gold-data benchmarks/MBE-exam-questions_distilled_dataset_cleaned.json \
	--split mbe/data/split.json \
	--all-data \
	--prompt-type baseline \
	--model Qwen/Qwen3-4B \
	--run-name mbe-qwen3-base \
	--output-root testing/baseline/huggingface/mbe
```

Choose the MBE prompt with `--prompt-type baseline` or `--prompt-type cot` and choose a separate run name to add the results in a different directory.
The evaluator loads the corresponding system prompt file automatically; use
`--prompt path/to/file.md` for a custom prompt instead. Do not pass both prompt
options. An explicit `--format-hint` adds answer-format instructions; use the
same prompt type and hint settings when comparing runs. The selected prompt
type, path, and SHA-256 are recorded in `run.json`. For example, replace
`--prompt-type baseline` with `--prompt-type cot` for an MBE CoT run.

The current baseline input omits the answer options. A letter may therefore be
parseable but cannot be reliably mapped to the correct option unless its
letter-to-answer mapping is inferable from the question itself.

Fine-tune on the training data, using validation loss to select the adapter:

```bash
python train_chameleon.py \
	--model Qwen/Qwen3-4B \
	--train-data mbe/data/train.jsonl \
	--valid-data mbe/data/valid.jsonl \
	--split mbe/data/split.json \
	--output-dir mbe/adapters_qwen3 \
	--max-steps 60 \
	--checkpoint-steps 15 \
	--learning-rate 1e-4 \
	--max-seq-length 4096
```

Upload the resulting PEFT adapter to Hugging Face. This uploads the adapter
configuration and weights; the base model is still loaded separately during
evaluation. Replace `YOUR_HF_USERNAME` with your account or organization:

```bash
hf auth login
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi

adapter_dir = Path("mbe/adapters_qwen3")
repo_id = "YOUR_HF_USERNAME/mbe-qwen3-lora"
config = adapter_dir / "adapter_config.json"
weights = next((adapter_dir / name for name in (
    "adapter_model.safetensors", "adapter_model.bin"
) if (adapter_dir / name).is_file()), None)

if not config.is_file() or weights is None:
    raise SystemExit(f"Adapter config/weights not found in {adapter_dir}")

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
for path in (config, weights):
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="model",
    )
print(f"Uploaded adapter to https://huggingface.co/{repo_id}")
PY
```

Evaluate the uploaded adapter on the same held-out test questions. You need
Hugging Face access to both the gated base model and the adapter repository:

```bash
python run_hf_mbe.py \
	--input mbe/data/test.jsonl \
	--split mbe/data/split.json \
	--model Qwen/Qwen3-4B \
	--adapter-path YOUR_HF_USERNAME/mbe-qwen3-lora \
	--run-name mbe-qwen3-adapter-hf \
	--output-root testing/baseline/huggingface/mbe
```

Both runs save `results.json`, `run.json`, and one Markdown file per question
under `testing/baseline/huggingface/mbe/<run-name>/`. The evaluator checks that
the test IDs match `split.json` and refuses to score training or validation
questions. Use validation results to choose training settings; do not use the
held-out test results for checkpoint selection.

### Accelerating Parallel Testing with `CUDA_VISIBLE_DEVICES`

On multi-GPU machines, running multiple test jobs without device isolation causes them to default to GPU `0`, leading to resource contention or CUDA Out-Of-Memory (OOM) errors while leaving secondary GPUs idle. Check this using `nvidia-smi` if any GPU is idle.

You can use the `CUDA_VISIBLE_DEVICES` environment variable to restrict each process to a specific GPU and run independent test runs in parallel:

```bash
# Terminal 1: Pin first run to GPU 0
export CUDA_VISIBLE_DEVICES=0
python <command>

# Terminal 2: Pin second run to GPU 1
export CUDA_VISIBLE_DEVICES=1
python <command>