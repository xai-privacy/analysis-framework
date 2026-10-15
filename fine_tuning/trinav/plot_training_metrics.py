#!/usr/bin/env python3
"""Export and plot metrics recorded in a Hugging Face Trainer state file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer-state",
        type=Path,
        default=Path(__file__).resolve().parent / "adapters_chameleon" / "trainer_state.json",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not args.trainer_state.is_file():
        raise SystemExit(
            f"Trainer state not found: {args.trainer_state}. "
            "Run train_chameleon.py first."
        )

    state = json.loads(args.trainer_state.read_text(encoding="utf-8"))
    rows = state.get("log_history", [])
    rows = [row for row in rows if isinstance(row, dict) and "step" in row]
    if not rows:
        raise SystemExit(f"No step metrics found in {args.trainer_state}")

    out = args.out or args.trainer_state.parent
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "training_metrics.csv"
    columns = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Plotting requires matplotlib. Install it with "
            "`python -m pip install matplotlib`. The metrics CSV was saved."
        ) from exc

    train_loss = [row for row in rows if "loss" in row and "eval_loss" not in row]
    eval_loss = [row for row in rows if "eval_loss" in row]
    learning_rate = [row for row in rows if "learning_rate" in row]

    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    if train_loss:
        axes[0].plot(
            [row["step"] for row in train_loss],
            [row["loss"] for row in train_loss],
            marker="o",
            label="Training loss",
        )
    if eval_loss:
        axes[0].plot(
            [row["step"] for row in eval_loss],
            [row["eval_loss"] for row in eval_loss],
            marker="s",
            label="Validation loss",
        )
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    if train_loss or eval_loss:
        axes[0].legend()

    if learning_rate:
        axes[1].plot(
            [row["step"] for row in learning_rate],
            [row["learning_rate"] for row in learning_rate],
            marker="o",
            color="tab:green",
        )
    axes[1].set_ylabel("Learning rate")
    axes[1].grid(True, alpha=0.3)

    epochs = [row for row in rows if "epoch" in row]
    if epochs:
        axes[2].plot(
            [row["step"] for row in epochs],
            [row["epoch"] for row in epochs],
            color="tab:orange",
        )
    axes[2].set_ylabel("Epoch")
    axes[2].set_xlabel("Training step")
    axes[2].grid(True, alpha=0.3)

    figure.suptitle("LEET-Arg QLoRA Training Metrics")
    figure.tight_layout()
    plot_path = out / "training_metrics.png"
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    print(f"Saved metrics CSV: {csv_path}")
    print(f"Saved metrics plot: {plot_path}")


if __name__ == "__main__":
    main()