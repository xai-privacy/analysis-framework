"""Shared Chameleon/CUDA helpers for Trinav LEET-Arg experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_DIR = HERE / "data"

HEADER_RE = re.compile(r"^\W*Statement\s*(\(\w\)|[①②③④⑤])", re.M)
VERDICT_WORD_RE = re.compile(r"Verdict\W*(INCORRECT|CORRECT)", re.I)


def parse_verdicts(output: str):
    """Read verdicts from the per-statement format used by Jun's run."""
    verdicts = {}
    headers = list(HEADER_RE.finditer(output))
    for header, following in zip(headers, headers[1:] + [None]):
        block = output[header.end():following.start() if following else len(output)]
        match = VERDICT_WORD_RE.search(block)
        if match and header.group(1) not in verdicts:
            verdicts[header.group(1)] = match.group(1).upper() == "CORRECT"
    return verdicts


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def validate_data_split(train_rows, valid_rows, split_path: Path):
    with split_path.open("r", encoding="utf-8") as handle:
        split = json.load(handle)

    train_ids = [str(row.get("id", "")) for row in train_rows]
    valid_ids = [str(row.get("id", "")) for row in valid_rows]
    if not all(train_ids) or not all(valid_ids):
        raise ValueError("Every training and validation record must have an id")
    if len(train_ids) != len(set(train_ids)) or len(valid_ids) != len(set(valid_ids)):
        raise ValueError("Duplicate IDs found in train or validation JSONL")
    if set(train_ids) != set(split.get("train", [])):
        raise ValueError("Training JSONL IDs do not match split.json")
    if set(valid_ids) != set(split.get("valid", [])):
        raise ValueError("Validation JSONL IDs do not match split.json")

    test_ids = set(map(str, split.get("test_ids", [])))
    if not test_ids:
        test_ids = {
            qid for qid, reason in split.get("excluded", {}).items()
            if reason == "final test question"
        }
    leaked = test_ids & (set(train_ids) | set(valid_ids))
    if leaked:
        raise ValueError(f"Final test questions found in train/valid data: {sorted(leaked)}")

    overlap = set(train_ids) & set(valid_ids)
    if overlap:
        raise ValueError(f"Train/validation overlap: {sorted(overlap)}")

    for row in train_rows + valid_rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or [m.get("role") for m in messages] != [
            "system", "user", "assistant"
        ]:
            raise ValueError(f"{row['id']}: expected system/user/assistant messages")
        if not all(isinstance(m.get("content"), str) and m["content"].strip()
                   for m in messages):
            raise ValueError(f"{row['id']}: message content cannot be blank")
        evaluation = row.get("evaluation")
        if not isinstance(evaluation, dict) or not isinstance(
            evaluation.get("gold_verdicts"), dict
        ) or not isinstance(evaluation.get("answer_verdicts"), dict):
            raise ValueError(f"{row['id']}: missing evaluation metadata; regenerate data")

    return split


def get_lora_target_modules(model) -> list[str]:
    """Dynamically determine appropriate LoRA target linear modules based on model architecture.

    Supports:
    - Phi-3, Phi-4, Phi-4-mini (fused qkv_proj and gate_up_proj with o_proj and down_proj)
    - LLaMA, Mistral, Qwen (split attention projections q, k, v, o and MLP gate, up, down)
    - Fallback: available linear projection layers detected in the model
    """
    # Extract the short names of all parameter-bearing leaf modules
    leaf_modules = {
        name.split(".")[-1]
        for name, mod in model.named_modules()
        if hasattr(mod, "weight")
    }

    # 1. Phi-3 / Phi-4 / Phi-4-mini family (fused attention and MLP projections)
    if "qkv_proj" in leaf_modules or "gate_up_proj" in leaf_modules:
        phi_targets = [
            m for m in ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
            if m in leaf_modules
        ]
        if phi_targets:
            return phi_targets

    # 2. LLaMA / Mistral / Qwen family (split attention and MLP projections)
    llama_targets = [
        m for m in [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
        if m in leaf_modules
    ]
    if llama_targets:
        return llama_targets

    # 3. Fallback: return standard projection names
    return ["q_proj", "v_proj"]


def load_cuda_model(model_id: str, adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if model_id == "mlx-community/Llama-3.2-3B-Instruct-4bit":
        raise ValueError(
            "Jun's exact checkpoint is MLX-quantized and cannot be loaded by this "
            "Transformers/bitsandbytes CUDA trainer. Use its original "
            "meta-llama/Llama-3.2-3B-Instruct checkpoint (or a local Transformers-format "
            "copy) as the compatible base; this is not the identical 4-bit checkpoint."
        )

    has_gpu = torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if has_gpu:
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization,
            torch_dtype=compute_dtype,
            device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True,
        )
    else:
        print("WARNING: CUDA GPU not detected. Falling back to CPU.")
        compute_dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )

    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, compute_dtype


__all__ = [
    "DATA_DIR",
    "HERE",
    "REPO",
    "get_lora_target_modules",
    "load_cuda_model",
    "parse_verdicts",
    "read_jsonl",
    "validate_data_split",
]