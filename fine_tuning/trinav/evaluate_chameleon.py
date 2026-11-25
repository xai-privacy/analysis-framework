#!/usr/bin/env python3
"""Run a base or QLoRA-adapted model on Trinav's validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    from .chameleon_utils import DATA_DIR, load_cuda_model, read_jsonl
    from .train_chameleon import DEFAULT_MODEL
except ImportError:
    from chameleon_utils import DATA_DIR, load_cuda_model, read_jsonl
    from train_chameleon import DEFAULT_MODEL

FORMAT_HINT = (
    "\n\nAnswer in exactly this format, one block per statement, using the "
    "statement's own label:\n\nStatement (a)\nReasoning: <your reasoning>\n"
    "Verdict: CORRECT or INCORRECT"
)


def save_results(path: Path, records: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def print_exchange(qid: str, prompt: str, response: str) -> None:
    print(f"\n===== {qid} | MODEL INPUT =====\n{prompt}")
    print(f"\n===== {qid} | MODEL RESPONSE =====")
    print(response, end="" if response.endswith("\n") else "\n", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Base model Hugging Face ID (e.g. Qwen/Qwen3-4B, microsoft/Phi-4-mini-instruct)",
    )
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--data", type=Path, default=DATA_DIR / "valid.jsonl")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--format-hint", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; choose another --out or pass --overwrite")

    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError(f"No validation rows found in {args.data}")
    if args.limit is not None:
        rows = rows[:args.limit]

    test_ids = set()
    split_path = DATA_DIR / "split.json"
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        test_ids = set(map(str, split.get("test_ids", [])))
        if not test_ids:
            test_ids = {
                qid for qid, reason in split.get("excluded", {}).items()
                if reason == "final test question"
            }
    leaked = test_ids & {str(row["id"]) for row in rows}
    if leaked:
        raise SystemExit(f"Refusing to evaluate final-test IDs: {sorted(leaked)}")

    for row in rows:
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or [message.get("role") for message in messages] != [
                "system", "user", "assistant"
            ]
            or any(
                not isinstance(message.get("content"), str)
                or not message["content"].strip()
                for message in messages
            )
        ):
            raise ValueError(f"{row.get('id')}: expected complete training messages")

    model, tokenizer, _ = load_cuda_model(
        args.model,
        str(args.adapter_path) if args.adapter_path else None,
    )
    model.eval()
    device = getattr(
        model,
        "device",
        torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        ),
    )
    records: dict[str, dict[str, str]] = {}

    for row in rows:
        qid = str(row["id"])
        messages = [dict(message) for message in row["messages"][:2]]
        if args.format_hint:
            messages[0]["content"] += FORMAT_HINT
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        response = tokenizer.decode(
            generated[0, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        records[qid] = {"prompt": rendered_prompt, "response": response}
        save_results(args.out, records)
        print_exchange(qid, rendered_prompt, response)
    return 0


if __name__ == "__main__":
    main()
