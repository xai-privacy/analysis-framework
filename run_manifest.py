# File: run_manifest.py
"""Capture everything needed to reproduce one benchmark invocation.

Each run of run_benchmark.py writes runs/<utc-timestamp>_<signature>/manifest.json
next to a console.log of the same run. Result records carry the matching run_id,
so a row in slm_results/*.json can always be traced back to the model revision,
generation config, prompt, dataset, code commit, and library versions that made
it -- which matters here because --resume and --overwrite let one result file
accumulate records from several sessions at different token budgets.

The fields chosen are the ones that silently change an experiment:
  * revision   -- HF repos move; "the model" is a moving target without a SHA
  * chat_template_sha256 -- a tokenizer update can reword every prompt
  * prompt_sha256, dataset_sha256 -- local edits to prompts.py / the benchmark
  * generation -- the merged dict actually passed to generate(), not the config file
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_RUNS_DIR = os.path.join(_REPO_ROOT, "runs")


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO_ROOT, *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_state():
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "describe": _git("describe", "--always", "--dirty"),
    }


def _package_versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "transformers", "tokenizers", "accelerate", "huggingface_hub"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def _device_info(torch_module, device):
    info = {"device": device, "platform": platform.platform()}
    if torch_module is None:
        return info
    try:
        if device == "cuda" and torch_module.cuda.is_available():
            props = torch_module.cuda.get_device_properties(0)
            major, minor = torch_module.cuda.get_device_capability(0)
            info.update(
                {
                    "gpu_name": props.name,
                    "compute_capability": f"sm_{major}{minor}",
                    "vram_gb": round(props.total_memory / 2 ** 30, 2),
                    "bf16_supported": bool(torch_module.cuda.is_bf16_supported()),
                    "cuda_version": torch_module.version.cuda,
                }
            )
    except Exception:
        pass
    return info


def resolve_revision(model_id, revision=None):
    """Resolve an HF model id to the commit SHA actually being used.

    Returns None when offline or the repo is gated/unreachable -- a missing
    revision is worth noting in the manifest, not worth failing a run over.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id, revision=revision).sha
    except Exception:
        return None


def build_manifest(
    run_id,
    model_id,
    model_cfg,
    gen_config,
    system_prompt,
    questions,
    device,
    torch_module,
    questions_path,
    cli_args,
    revision=None,
    tokenizer=None,
):
    """Assemble the manifest dict. Pure aside from git/HF/filesystem probing."""
    chat_template = getattr(tokenizer, "chat_template", None) if tokenizer else None
    return {
        "run_id": run_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "finished_utc": None,
        "model": {
            "model_id": model_id,
            "revision_requested": revision,
            "revision_resolved": resolve_revision(model_id, revision),
            "torch_dtype": model_cfg.get("torch_dtype"),
            "trust_remote_code": model_cfg.get("trust_remote_code", False),
            "seed": model_cfg.get("seed", 0),
            "reasoning": model_cfg.get("reasoning"),
            "chat_template_sha256": _sha256_text(chat_template) if chat_template else None,
        },
        "generation": dict(gen_config),
        "prompt": {
            "system_prompt": system_prompt,
            "sha256": _sha256_text(system_prompt),
        },
        "dataset": {
            "path": os.path.relpath(questions_path, _REPO_ROOT),
            "sha256": _sha256_file(questions_path),
            "question_count": len(questions),
            "question_ids": [question["id"] for question in questions],
        },
        "code": _git_state(),
        "environment": {
            "packages": _package_versions(),
            **_device_info(torch_module, device),
        },
        "cli": cli_args,
        "totals": None,
    }


def run_dir(run_id):
    return os.path.join(_RUNS_DIR, run_id)


def new_run_id(model_id):
    signature = model_id.replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{signature}"


def write_manifest(run_id, manifest):
    directory = run_dir(run_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def finalize(run_id, manifest, results_for_run):
    """Stamp end time and aggregate counters, then rewrite the manifest.

    Called even on an interrupted run, so a manifest with finished_utc set but
    fewer questions than the dataset is the signature of a partial run.
    """
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    stop_reasons = {}
    completion_tokens = []
    seconds = 0.0
    parsed = 0
    for record in results_for_run:
        reason = record.get("stop_reason") or "unknown"
        stop_reasons[reason] = stop_reasons.get(reason, 0) + 1
        if record.get("completion_tokens") is not None:
            completion_tokens.append(record["completion_tokens"])
        if record.get("gen_seconds"):
            seconds += record["gen_seconds"]
        if record.get("model_answer") is not None:
            parsed += 1
    manifest["totals"] = {
        "questions_run": len(results_for_run),
        "parsed_answers": parsed,
        "stop_reasons": stop_reasons,
        "completion_tokens_sum": sum(completion_tokens) or None,
        "gen_seconds_sum": round(seconds, 2) or None,
        "tokens_per_second": round(sum(completion_tokens) / seconds, 2) if seconds else None,
    }
    write_manifest(run_id, manifest)
    return manifest


def append_index(manifest):
    """One JSONL line per run, so runs can be compared without walking directories."""
    os.makedirs(_RUNS_DIR, exist_ok=True)
    summary = {
        "run_id": manifest["run_id"],
        "started_utc": manifest["started_utc"],
        "finished_utc": manifest["finished_utc"],
        "model_id": manifest["model"]["model_id"],
        "revision": manifest["model"]["revision_resolved"],
        "max_new_tokens": manifest["generation"].get("max_new_tokens"),
        "questions_run": (manifest.get("totals") or {}).get("questions_run"),
        "parsed_answers": (manifest.get("totals") or {}).get("parsed_answers"),
        "stop_reasons": (manifest.get("totals") or {}).get("stop_reasons"),
        "commit": manifest["code"]["commit"],
        "dirty": manifest["code"]["dirty"],
        "gpu": manifest["environment"].get("gpu_name", manifest["environment"].get("device")),
    }
    with open(os.path.join(_RUNS_DIR, "index.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False) + "\n")


class Tee:
    """Mirror a stream to a log file so stderr warnings survive the tmux scrollback.

    run_benchmark.py's config warning and reasoning-tag warnings both go to
    stderr and are otherwise lost.
    """

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data):
        self._stream.write(data)
        self._handle.write(data)
        self._handle.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        return self._stream.isatty()


def open_console_log(run_id):
    """Return (handle, restore_callable) with stdout/stderr teed into console.log."""
    directory = run_dir(run_id)
    os.makedirs(directory, exist_ok=True)
    handle = open(os.path.join(directory, "console.log"), "a", encoding="utf-8")
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(saved_stdout, handle)
    sys.stderr = Tee(saved_stderr, handle)

    def restore():
        sys.stdout, sys.stderr = saved_stdout, saved_stderr
        handle.close()

    return handle, restore
