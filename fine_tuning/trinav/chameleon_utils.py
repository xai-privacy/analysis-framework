"""Shared Chameleon/CUDA helpers for Trinav LEET-Arg experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_DIR = HERE / "data"

sys.path.insert(0, str(REPO))
from fine_tuning.jun.leet_arg_format import (  # noqa: E402
    answer_from_verdicts,
    gold_verdicts,
    load_dataset,
    load_final_test_ids,
    parse_verdicts,
)


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

    test_ids = load_final_test_ids()
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

    return split


def load_cuda_model(model_id: str, adapter_path: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not detected. Run this script in a GPU-backed Chameleon instance, "
            "not the local CPU-only environment."
        )

    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map={"": torch.cuda.current_device()},
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
    "answer_from_verdicts",
    "gold_verdicts",
    "load_cuda_model",
    "load_dataset",
    "load_final_test_ids",
    "parse_verdicts",
    "read_jsonl",
    "validate_data_split",
]