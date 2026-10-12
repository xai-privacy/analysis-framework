#!/usr/bin/env python3
"""QLoRA fine-tuning for the Trinav LEET-Arg rationale dataset on CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    from .chameleon_utils import (
        DATA_DIR,
        HERE,
        load_cuda_model,
        read_jsonl,
        validate_data_split,
    )
except ImportError:
    from chameleon_utils import (
        DATA_DIR,
        HERE,
        load_cuda_model,
        read_jsonl,
        validate_data_split,
    )

# This CUDA workflow uses the compatible
# original Hugging Face model weights and must not be described as identical.
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


class MaskedChatDataset(Dataset):
    """Tokenize chats and compute loss only on the assistant completion."""

    def __init__(self, rows, tokenizer, max_seq_length):
        self.examples = []
        for row in rows:
            messages = row["messages"]
            prefix_ids = tokenizer.apply_chat_template(
                messages[:2], tokenize=True, add_generation_prompt=True
            )
            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
            if hasattr(prefix_ids, "tolist"):
                prefix_ids = prefix_ids.tolist()
            if hasattr(full_ids, "tolist"):
                full_ids = full_ids.tolist()
            if prefix_ids and isinstance(prefix_ids[0], list):
                prefix_ids = prefix_ids[0]
            if full_ids and isinstance(full_ids[0], list):
                full_ids = full_ids[0]
            if full_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError(
                    f"{row['id']}: tokenizer chat template does not preserve the "
                    "generation prompt as a prefix; refusing to mask the wrong tokens"
                )
            if len(full_ids) > max_seq_length:
                raise ValueError(
                    f"{row['id']}: {len(full_ids)} tokens exceeds --max-seq-length "
                    f"{max_seq_length}; increase the limit rather than silently truncating"
                )
            if len(full_ids) <= len(prefix_ids):
                raise ValueError(f"{row['id']}: assistant target tokenized as empty")

            labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
            self.examples.append({
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return {key: torch.tensor(value, dtype=torch.long) for key, value in example.items()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DATA_DIR / "train.jsonl")
    parser.add_argument("--valid-data", type=Path, default=DATA_DIR / "valid.jsonl")
    parser.add_argument("--split", type=Path, default=DATA_DIR / "split.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "adapters_chameleon")
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_steps < 1 or args.eval_steps < 1:
        raise SystemExit("--max-steps and --eval-steps must be positive")

    train_rows = read_jsonl(args.train_data)
    valid_rows = read_jsonl(args.valid_data)
    validate_data_split(train_rows, valid_rows, args.split)

    model, tokenizer, compute_dtype = load_cuda_model(args.model)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    train_dataset = MaskedChatDataset(train_rows, tokenizer, args.max_seq_length)
    valid_dataset = MaskedChatDataset(valid_rows, tokenizer, args.max_seq_length)

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    model = get_peft_model(model, LoraConfig(
        r=8,
        lora_alpha=20,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ))
    model.print_trainable_parameters()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        save_safetensors=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
        processing_class=tokenizer,
    )

    print(
        f"Training {len(train_dataset)} examples; validating on {len(valid_dataset)}; "
        f"CUDA dtype={compute_dtype}; max_steps={args.max_steps}"
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    print("Saved best adapter to", args.output_dir.resolve())


if __name__ == "__main__":
    main()