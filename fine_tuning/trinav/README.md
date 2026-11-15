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
adapters. Jun's original checkpoint,
`mlx-community/Llama-3.2-3B-Instruct-4bit`, is MLX-format and cannot be loaded
by this Transformers/bitsandbytes CUDA workflow. The commands below use the
original `meta-llama/Llama-3.2-3B-Instruct` weights, quantized to NF4 at load
time; these are the same model family, but not the exact same quantized
checkpoint.

Accept Meta's model license on Hugging Face before starting. The model is
downloaded by Transformers when a run starts; no model weights need to be copied
to the VM.

### Set up the Chameleon VM

From `analysis-framework/fine_tuning/chameleon` on your local machine, upload
the Trinav folder. Replace the key and IP with your own:

```bash
ssh -i ../my-chi-key.pem cc@<public-ip> 'mkdir -p ~/xai'
scp -i ../my-chi-key.pem -r ../trinav cc@<public-ip>:~/xai/
```

Log in, create the environment, and install the pinned dependencies. The
requirements pin PyTorch to the CUDA 12.6 wheel with P100 (`sm_60`) support, so
the incompatible image-provided PyTorch build is replaced inside this venv:

```bash
ssh -i ../my-chi-key.pem cc@<public-ip>
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

The final test runner needs the benchmark questions and final-test ID list on
the VM. From the repository root on your local machine, copy them once:

```bash
scp -i <private-key.pem> \
	benchmarks/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	cc@<public-ip>:~/xai/trinav/data/
scp -i <private-key.pem> \
	testing/final_testing/test_set/LEET_Arg_Questions_Test_Set.json \
	cc@<public-ip>:~/xai/trinav/data/
```

### 1. Baseline testing

Run the base model first on the held-out final test IDs. This records the
baseline for comparison with the adapter. The runner refuses to evaluate
training or validation IDs. Do not use final-test results to choose training
steps or adapters; use validation results for those decisions.

```bash
cd ~/xai/trinav
source .venv/bin/activate
python run_hf_questions.py \
	--input data/LEET_Arg_Questions_cleaned_and_rationale_by_statement.json \
	--test-ids data/LEET_Arg_Questions_Test_Set.json \
	--split data/split.json \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--run-name llama32-base \
	--output-root testing/baseline/huggingface
```

### 2. Fine-tuning

Train for 180 steps, evaluating and saving every 20 steps. Validation loss is
used during training to select the best checkpoint. The final adapter is saved
to `adapters_chameleon/`.

```bash
python train_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--train-data data/train.jsonl \
	--valid-data data/valid.jsonl \
	--split data/split.json \
	--output-dir adapters_chameleon \
	--max-steps 180 \
	--eval-steps 20 \
	--learning-rate 1e-4 \
	--max-seq-length 4096
```

The trainer uses batch size 1, LoRA rank 8, and a constant learning rate. If an
example exceeds the sequence length, it stops with that example's ID rather
than truncating the rationale. To plot the training and validation metrics:

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
repo_id = "YOUR_HF_USERNAME/leetarg-llama32-lora"
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
	--model meta-llama/Llama-3.2-3B-Instruct \
	--adapter-path YOUR_HF_USERNAME/leetarg-llama32-lora \
	--run-name llama32-adapter-hf \
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

## MBE Fine-Tuning Dataset

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

Regenerate the split from the repository root:

```bash
python fine_tuning/trinav/mbe/prepare_mbe_data.py
```

The script writes `train.jsonl`, `valid.jsonl`, `test.jsonl`, `split.json`, and
ID lists under `fine_tuning/trinav/mbe/data/`. The same trainer can fine-tune on
the MBE chat data by pointing it at the MBE train, validation, and split files:

```bash
python fine_tuning/trinav/train_chameleon.py \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--train-data fine_tuning/trinav/mbe/data/train.jsonl \
	--valid-data fine_tuning/trinav/mbe/data/valid.jsonl \
	--split fine_tuning/trinav/mbe/data/split.json \
	--output-dir fine_tuning/trinav/mbe/adapters_llama32 \
	--max-steps 180 \
	--eval-steps 20 \
	--learning-rate 1e-4 \
	--max-seq-length 4096
```

To train on Chameleon, copy the `mbe/` folder to the VM as well as the shared
trainer scripts. Use paths relative to `~/xai/trinav`, for example `mbe/data/`
and `mbe/adapters_llama32/`. Keep `test.jsonl` out of training and validation.