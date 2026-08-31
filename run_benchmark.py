import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from prompts import get_system_prompt

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover - optional dependency path
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
except Exception:  # pragma: no cover - optional dependency path
    hf_hub_download = None
    EntryNotFoundError = Exception

_MODEL_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_configs")
_QUESTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "benchmarks",
    # The by-statement file, matching run_claude_benchmark.py, run_openai_benchmark.py
    # and results/evaluate_results.py. Commit f303958 moved those three and missed this
    # one, which left SLM rows carrying original_rationale as a flat string while API
    # rows carry the per-statement dict (90 of 93 questions differ). Same ids, same 93
    # questions, so nothing here breaks -- only the rationale shape.
    "LEET_Arg_Questions_cleaned_and_rationale_by_statement.json",
)
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
_FALLBACK_CONFIG_NAME = "meta-llama_Llama-3.2-1B-Instruct.json"

# Tag families used by reasoning models to delimit their thinking block, in the
# two delimiter styles that appear on issue #18's model list: angle brackets
# (<think> for DeepSeek-R1-Distill, LiquidAI, Phi-4-reasoning-plus) and square
# brackets ([THINK] for the Ministral-3 Reasoning models). Kept as a name
# alternation because the close form is mechanically derived from the open one.
_REASONING_TAG_NAMES = ("think", "thought", "reasoning", "analysis")
_REASONING_OPEN_PATTERN = re.compile(
    r"<(?P<angle>" + "|".join(_REASONING_TAG_NAMES) + r")>"
    r"|\[(?P<bracket>" + "|".join(_REASONING_TAG_NAMES) + r")\]",
    re.IGNORECASE,
)

# max_new_tokens is a caller-side runtime parameter, not a property of the
# checkpoint, so no HF repo publishes it (absent from generation_config.json in
# all 8 models checked). Any value here is a placeholder for a first
# characterization run, never an answer.
#
# Deliberately generous rather than typical. A cap that truncates tells you
# nothing except that it truncated, and for a thinking model a response cut
# before its closing tag has no parseable answer at all, so the run reads as a
# reasoning failure instead of a budget problem. 8192 is what LFM2.5-Thinking
# needed to finish all 93 questions on eos. Non-thinking models stop well short
# of it and pay nothing for the headroom. Narrow it afterwards from measured
# completion lengths; stop_reason shows whether the headroom was used.
_PROVISIONAL_MAX_NEW_TOKENS = 8192


def _config_file_resolver(model_id):
    """Return a callable that maps a config filename to a readable local path.

    Two sources, one interface. A local directory (as produced by
    `hf download --local-dir`) already contains every file this function wants,
    so it is read straight off disk -- no network, no auth, and it works for
    gated models the Hub would refuse. Anything else is treated as a Hub repo
    id. Both raise EntryNotFoundError for a missing file so callers can use one
    except clause.
    """
    if os.path.isdir(model_id):
        def resolve(filename):
            path = os.path.join(model_id, filename)
            if not os.path.isfile(path):
                raise EntryNotFoundError(f"{filename} not present in {model_id}")
            return path
        return resolve
    return lambda filename: hf_hub_download(model_id, filename)


def _describe_fetch_failure(model_id, exc):
    """Turn a huggingface_hub exception into a message that names the actual
    cause. Every failure mode below otherwise surfaces as the same opaque HTTP
    error, 401 without a token and 404 with one, so a gated repo, a nonexistent
    repo and a mistyped local path are indistinguishable. The 401 in particular
    sends you looking for a token problem that may not exist."""
    name = type(exc).__name__
    if name == "HFValidationError":
        return (
            f"{model_id!r} is not a valid Hub repo id. If it is meant to be a local "
            "directory, check the path exists -- local directories are read directly "
            "and never fetched."
        )
    if name in ("GatedRepoError", "RepositoryNotFoundError"):
        token_hint = ""
        if _hf_token() is None:
            token_hint = (
                " No Hugging Face token was found, so gated repos will always fail here; "
                "run `hf auth login` or set HF_TOKEN."
            )
        path_hint = ""
        if os.sep in model_id or "/" in model_id:
            path_hint = (
                f" If {model_id!r} was meant to be a local directory, it does not exist "
                "relative to the current working directory -- check the path."
            )
        return (
            f"{model_id!r} could not be read from the Hub -- it is gated, private, or "
            f"does not exist.{token_hint}{path_hint}"
        )
    if name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled"):
        return f"No network access while fetching {model_id!r}, and it is not in the local cache."
    return f"Failed to fetch config for {model_id!r}: {name}: {exc}"


def _hf_token():
    """Best-effort lookup of a configured HF token (env var or `hf auth login`
    credentials file). Returns None when huggingface_hub is unavailable."""
    try:
        from huggingface_hub import get_token
    except Exception:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    try:
        return get_token()
    except Exception:
        return None


def _detect_reasoning_tags(resolve):
    """Look for a reasoning-block tag pair declared in the tokenizer's files.

    Two independent sources, unioned, because neither alone is sufficient:
    the chat template (DeepSeek-R1 and Granite have the tags only there) and
    added_tokens_decoder (LiquidAI, Qwen3 and SmolLM3 register them as added
    tokens). The template itself lives in either tokenizer_config.json's
    "chat_template" field or a standalone chat_template.jinja -- LiquidAI uses
    only the latter, so both are checked.

    This is best-effort and cannot be made complete. Phi-4-mini-reasoning
    declares nothing anywhere and emits <think> as learned behavior; Magistral
    keeps [THINK] in tekken.json with no template file at all. Those are caught
    at runtime by detect_undeclared_reasoning_tag(), not here. Returns a
    (open_tag, close_tag) pair or None.
    """
    haystacks = []

    try:
        with open(resolve("tokenizer_config.json"), "r", encoding="utf-8") as f:
            tokenizer_cfg = json.load(f)
    except EntryNotFoundError:
        tokenizer_cfg = {}
    except Exception as exc:
        # Non-fatal: tag detection is best-effort and must not block the fetch.
        print(f"[Config warning] Could not read tokenizer_config.json: {exc}", file=sys.stderr)
        tokenizer_cfg = {}

    template_field = tokenizer_cfg.get("chat_template", "")
    if isinstance(template_field, list):
        # Some repos store multiple named templates as a list of dicts.
        template_field = " ".join(
            str(entry.get("template", "")) for entry in template_field if isinstance(entry, dict)
        )
    haystacks.append(str(template_field))

    added = tokenizer_cfg.get("added_tokens_decoder")
    if isinstance(added, dict):
        haystacks.extend(
            str(entry.get("content", ""))
            for entry in added.values()
            if isinstance(entry, dict)
        )

    try:
        with open(resolve("chat_template.jinja"), "r", encoding="utf-8") as f:
            haystacks.append(f.read())
    except EntryNotFoundError:
        pass
    except Exception as exc:
        print(f"[Config warning] Could not read chat_template.jinja: {exc}", file=sys.stderr)

    combined = "\n".join(haystacks)
    for match in _REASONING_OPEN_PATTERN.finditer(combined):
        # Case is taken from the match, not normalized: Mistral writes [THINK]
        # in caps and the tag must be reproduced exactly to match generated text.
        name = match.group("angle")
        open_tag, close_tag = (f"<{name}>", f"</{name}>") if name else (
            f"[{match.group('bracket')}]", f"[/{match.group('bracket')}]")
        if close_tag.lower() in combined.lower():
            return open_tag, close_tag
    return None


def _fetch_config_from_hf(model_id):
    """Build a model config from a checkpoint's own config.json and
    generation_config.json, read either from a local directory or the HF Hub.

    Two rules govern what lands in the result. Values the source actually states
    (dtype, auto_map, reasoning tags) are taken from it. Values the source is
    silent about -- max_new_tokens above all -- are marked provisional in
    "_provisional" so _load_model_config() keeps warning about them instead of
    presenting a placeholder as a finished config. Decoding is forced to greedy
    regardless of what the repo specifies.

    Returns None on failure so the caller can fall back.
    """
    is_local = os.path.isdir(model_id)
    if hf_hub_download is None and not is_local:
        print(
            "[Config warning] huggingface_hub is not available; cannot auto-fetch config.",
            file=sys.stderr,
        )
        return None

    resolve = _config_file_resolver(model_id)

    cfg = {
        "model_id": model_id,
        "torch_dtype": "float16",
        "trust_remote_code": False,
        "seed": 0,
        "generation": {
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": _PROVISIONAL_MAX_NEW_TOKENS,
        },
    }
    provisional = ["max_new_tokens"]

    # Only the two real config files count as evidence that this checkpoint was
    # actually resolved. The tokenizer files are inspected for reasoning tags
    # further down and must not, on their own, make an all-defaults config look
    # like a successful fetch.
    fetched_anything = False

    try:
        with open(resolve("config.json"), "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
        fetched_anything = True

        # config.json's key was renamed torch_dtype -> dtype in transformers 4.56.
        dtype = model_cfg.get("torch_dtype") or model_cfg.get("dtype")
        if dtype in ("float16", "bfloat16"):
            cfg["torch_dtype"] = dtype
        elif dtype is not None:
            cfg["torch_dtype"] = "auto"
        # Upstream dtype is what the weights were saved in, not necessarily what
        # runs well here -- bfloat16 is poorly supported on MPS, which is why the
        # committed Llama config pins float16 against the Hub's bfloat16.
        provisional.append("torch_dtype")

        if "auto_map" in model_cfg:
            cfg["trust_remote_code"] = True
    except EntryNotFoundError:
        pass
    except Exception as exc:
        print(f"[Config warning] {_describe_fetch_failure(model_id, exc)}", file=sys.stderr)
        return None

    try:
        with open(resolve("generation_config.json"), "r", encoding="utf-8") as f:
            gen_cfg = json.load(f)
        fetched_anything = True

        # Only length controls are taken from the repo; sampling behavior is
        # forced below. In practice max_new_tokens is never present -- kept for
        # the rare repo that does publish one, which then stops being a guess.
        if "max_new_tokens" in gen_cfg:
            cfg["generation"]["max_new_tokens"] = gen_cfg["max_new_tokens"]
            provisional.remove("max_new_tokens")
    except EntryNotFoundError:
        pass
    except Exception as exc:
        print(f"[Config warning] {_describe_fetch_failure(model_id, exc)}", file=sys.stderr)
        return None

    if not fetched_anything:
        print(
            f"[Config warning] Neither config.json nor generation_config.json found for {model_id}.",
            file=sys.stderr,
        )
        return None

    # Forced, non-negotiable: greedy decoding for auto-fetched configs.
    cfg["generation"]["do_sample"] = False
    cfg["generation"]["temperature"] = 0.0

    tags = _detect_reasoning_tags(resolve)
    cfg["reasoning"] = {"open_tag": tags[0], "close_tag": tags[1]} if tags else None

    cfg["_provisional"] = provisional
    return cfg


def _warn_provisional(model_id, cfg, config_path):
    """Warn, on every load, about fields no source could supply. Deliberately
    not silenced by the write to model_configs/: the whole failure mode this
    guards against is a placeholder becoming permanent because it got cached
    once and never questioned again."""
    provisional = cfg.get("_provisional")
    if not provisional:
        return
    if "max_new_tokens" in provisional:
        cap = cfg.get("generation", {}).get("max_new_tokens")
        print(
            f"[Config warning] {model_id}: max_new_tokens={cap} is a placeholder, not a "
            "measured value -- no HF repo publishes this field. It is set high on purpose "
            "so a first run is not truncated. Narrow it from that run's completion lengths "
            "and stop_reason counts.",
            file=sys.stderr,
        )
    other = [f for f in provisional if f != "max_new_tokens"]
    if other:
        print(
            f"[Config warning] {model_id}: {', '.join(other)} taken from the checkpoint's own "
            "metadata, which may not match what runs well on this machine.",
            file=sys.stderr,
        )
    print(
        f'[Config] Remove "_provisional" from {config_path} once these are confirmed.',
        file=sys.stderr,
    )


def _load_model_config(model_id):
    """Resolve a model config with a three-tier strategy:
    1. Local model_configs/<sanitized_model_id>.json, if present.
    2. Auto-derived from the checkpoint's own files -- read from a local
       directory when --model points at one, otherwise fetched from the Hub --
       with decoding forced to greedy. Cached to disk so later runs skip it,
       but any provisional field keeps warning until a human confirms it.
    3. Fall back to the Llama default config, with an explicit warning.
    """
    sanitized = model_id.replace("/", "_").strip("._")
    config_path = os.path.join(_MODEL_CONFIGS_DIR, f"{sanitized}.json")

    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        _warn_provisional(model_id, cfg, config_path)
        return cfg

    source = "local directory" if os.path.isdir(model_id) else "the Hugging Face Hub"
    print(
        f"[Config warning] No config found at {config_path} -- deriving {model_id}'s "
        f"config from {source}...",
        file=sys.stderr,
    )
    cfg = _fetch_config_from_hf(model_id)
    if cfg is not None:
        try:
            os.makedirs(_MODEL_CONFIGS_DIR, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(
                f"[Config] Auto-derived config for {model_id} saved to {config_path} "
                "(do_sample=False, temperature=0.0 forced).",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[Config warning] Derived config but failed to cache it: {exc}", file=sys.stderr)
        _warn_provisional(model_id, cfg, config_path)
        return cfg

    # Fall back to Llama config as default
    fallback_path = os.path.join(_MODEL_CONFIGS_DIR, _FALLBACK_CONFIG_NAME)
    with open(fallback_path, "r", encoding="utf-8") as f:
        fallback = json.load(f)
    print(
        f"[Config warning] Could not derive a config for {model_id} -- falling back to "
        "meta-llama/Llama-3.2-1B-Instruct's settings (non-thinking, "
        f"{fallback.get('torch_dtype')}, do_sample=False, "
        f"max_new_tokens={fallback.get('generation', {}).get('max_new_tokens')}). These are "
        "another model's values and are almost certainly wrong here. Add a "
        f"model_configs/{sanitized}.json with this model's actual settings before trusting "
        "any results from this run.",
        file=sys.stderr,
    )
    return fallback


def parse_model_response(response, open_tag="<think>", close_tag="</think>"):
    """Extract the answer marker and retain the remaining response as rationale.

    The reasoning block, where there is one, is cut away before the answer is
    searched for. Skipping that step lets the regex match a candidate the model
    proposed and then talked itself out of, recording a discarded guess as the
    final answer -- a wrong score that looks exactly like a right one.

    The tags come from the model's config instead of being hardcoded, because
    neither the delimiter style nor the presence of the opener is universal:

    - Mistral's reasoning models delimit with [THINK]...[/THINK], not <think>.
    - DeepSeek-R1-Distill's chat template appends "<think>" to the *prompt*
      (`{{'<|Assistant|><think>\\n'}}`), so generation begins already inside the
      block and the opener never appears in the decoded completion -- only the
      closer does.

    Presence of the *closer* therefore decides whether a block was emitted; the
    opener is only consulted to tell an unterminated block from an absent one.
    """
    search_text = response
    offset = 0

    if close_tag:
        # rfind, not find: a model that emits several reasoning blocks puts its
        # final answer after the last one, and everything earlier is working.
        close_idx = response.lower().rfind(close_tag.lower())
        if close_idx == -1:
            # A model whose config declares reasoning tags but whose output has
            # no closer either ran out of budget mid-thought or never finished
            # the block. Either way the text is reasoning in progress, not a
            # conclusion, so no answer is reported. Deliberately strict: a
            # missing answer shows up as unparseable in the summary, whereas a
            # mid-reasoning guess would silently corrupt the score.
            return {"model_answer": None, "model_rationale": response.strip()}
        offset = close_idx + len(close_tag)
        search_text = response[offset:]

    answer_match = re.search(
        r"\bAnswer\s*-\s*\{?\s*(?!choice\b)([0-9]+\b|[A-Ea-e]\b|[①②③④⑤])",
        search_text,
        re.IGNORECASE,
    )
    if answer_match is None:
        return {"model_answer": None, "model_rationale": response.strip()}

    start, end = offset + answer_match.start(), offset + answer_match.end()
    rationale = (response[:start] + response[end:]).strip()
    return {"model_answer": answer_match.group(1).strip(), "model_rationale": rationale}


_LEADING_TAG_PATTERN = re.compile(r"^\s*(<[A-Za-z_]+>|\[[A-Za-z_]+\])")


def detect_undeclared_reasoning_tag(response, declared_open_tag=None):
    """Flag a paired open/close tag opening the response that isn't the model's
    declared reasoning tag. Returns the tag string if flagged, else None.

    Unpaired tags (e.g. this dataset's own <statements>/<choices> markers, which
    never appear with a matching close form anywhere in the questions) are not
    flagged -- only a real open+close pair at the start of generation looks like
    reasoning-block syntax the parser should know about.
    """
    match = _LEADING_TAG_PATTERN.match(response)
    if not match:
        return None
    opener = match.group(1)
    name = opener[1:-1]
    closer = f"</{name}>" if opener.startswith("<") else f"[/{name}]"
    if closer not in response:
        return None
    if declared_open_tag and opener.lower() == declared_open_tag.lower():
        return None
    return opener


# Every row this runner writes carries run_id 1. Decoding is greedy with a fixed
# seed, so repeating a question reproduces the same tokens and there is nothing
# to average over. The field exists so these files share a shape with the API
# runners, which do sample and do support --runs.
_RUN_ID = 1

# Only the fields that change what the model emits. Everything else in a config
# ("_provisional" bookkeeping above all) can be edited freely without
# invalidating rows already on disk, and must not trip the drift check.
_CONFIG_FINGERPRINT_FIELDS = ("model_id", "torch_dtype", "seed", "generation", "reasoning")


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config_fingerprint(model_cfg):
    subset = {key: model_cfg.get(key) for key in _CONFIG_FINGERPRINT_FIELDS}
    return _sha256_text(json.dumps(subset, sort_keys=True, ensure_ascii=False))[:12]


def _git_state():
    """The commit this run executed at, and whether the tree was dirty.

    Both halves are needed: a commit alone describes the code only if nothing
    was edited on top of it, and on a Colab VM that is worth recording rather
    than assuming.
    """
    repo = os.path.dirname(os.path.abspath(__file__))

    def git(*args):
        return subprocess.run(args, cwd=repo, capture_output=True, text=True, timeout=10)

    try:
        head = git("git", "rev-parse", "HEAD")
        if head.returncode != 0:
            return {"commit": None, "dirty": None}
        status = git("git", "status", "--porcelain")
        return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def _versions():
    versions = {"python": platform.python_version(), "torch": getattr(torch, "__version__", None)}
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception:
        versions["transformers"] = None
    return versions


def _device_name(device):
    try:
        if device == "cuda":
            return torch.cuda.get_device_name(0)
        if device == "mps":
            return f"Apple Silicon ({platform.machine()})"
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _run_uid(model_id):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', model_id).strip('_')}"


def _write_run_metadata(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def generate_hf_response(model, tokenizer, user_content, device, system_prompt, gen_config):
    """Generate one completion and report how it ended.

    Returns a dict, not a bare string, because the text alone cannot tell a
    model that finished from one cut off at max_new_tokens -- and for a thinking
    model those are indistinguishable in the saved output while meaning opposite
    things: a response truncated before its closing tag has no parseable answer,
    so a budget problem is recorded as a reasoning failure. transformers'
    generate() reports nothing equivalent to the hosted APIs' stop_reason, so it
    is derived here.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    prompt_len = int(inputs["input_ids"].shape[1])

    generation_kwargs = dict(gen_config)

    try:
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
    except Exception as exc:
        print(f"[Generation Error]: {exc}", file=sys.stderr)
        return {
            "text": "",
            "input_tokens": prompt_len,
            "output_tokens": 0,
            "stop_reason": None,
            "error": f"generation_failed: {exc}",
        }

    completion = output_tokens[0][prompt_len:]
    completion_len = int(completion.shape[0])
    cap = generation_kwargs.get("max_new_tokens")

    # eos is tested before the cap, never after. A model that emits its stop
    # token on the last permitted step finished normally, and reporting that as
    # "length" would invent a truncation that did not happen -- which is exactly
    # the misreading this field exists to prevent.
    eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        eos_ids = set()
    elif isinstance(eos_ids, int):
        eos_ids = {eos_ids}
    else:
        eos_ids = set(eos_ids)

    if completion_len and int(completion[-1]) in eos_ids:
        stop_reason = "eos"
    elif cap is not None and completion_len >= cap:
        stop_reason = "length"
    else:
        # Neither the stop token nor the cap: some other StoppingCriteria fired.
        stop_reason = "stop"

    text = tokenizer.decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "text": text,
        "input_tokens": prompt_len,
        "output_tokens": completion_len,
        "stop_reason": stop_reason,
        "error": None,
    }


def _result_path(model_id):
    signature = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
    return os.path.join(_RESULTS_DIR, f"{signature}.json")


def _load_questions(year=None, limit=None):
    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if year is not None:
        year_prefix = f"{year}_"
        questions = [
            question for question in questions
            if str(question.get("id", "")).startswith(year_prefix)
        ]
    if limit is not None:
        questions = questions[:limit]
    return questions


def _completed_ids(results):
    """Question ids that do not need to be asked again.

    A row carrying an error is not counted: those failures are transient (OOM, a
    dropped connection, a CUDA hiccup) and a rerun can genuinely succeed.

    A row truncated at max_new_tokens IS counted as done. Decoding is greedy
    with a fixed seed, so rerunning a truncated question reproduces the same
    truncated text token for token -- retrying spends GPU time to learn nothing
    and never terminates. Only raising the cap changes that outcome, and raising
    the cap is a config change, which _check_config_drift() catches instead.
    """
    completed = set()
    for row in results:
        if row.get("error") is not None:
            continue
        qid = row.get("id")
        if qid is not None:
            completed.add((qid, row.get("run_id", _RUN_ID)))
    return completed


def _check_config_drift(results, fingerprint, model_id):
    """Refuse to append rows generated under different settings than the ones on disk.

    The failure this prevents is silent. Raise max_new_tokens after seeing
    truncation, rerun without --overwrite, and the finished questions are
    skipped as complete while the remaining ones run under the new budget. The
    file then holds two experiments and nothing in it says which row belongs to
    which.
    """
    seen = {row.get("config_sha") for row in results if row.get("config_sha")}
    stale = seen - {fingerprint}
    if not stale:
        return
    raise SystemExit(
        f"[Config drift] {model_id}: rows already in the results file were produced with "
        f"model config {'/'.join(sorted(stale))}, but the current config fingerprints as "
        f"{fingerprint}. Appending would mix two different generation settings in one "
        "file. Rerun with --overwrite to discard the old rows, or restore the previous "
        f"model_configs/ entry to keep them."
    )


def _percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(round(fraction * (len(sorted_values) - 1))))
    return sorted_values[index]


def _summarize_run(results, model_id, cap):
    """Fold the whole result file into the numbers that decide the token budget.

    Deliberately computed over every row in the file, not just this run's: the
    question being answered is "what does this model need on this benchmark",
    and a resumed run holds half the evidence.
    """
    usable = [row for row in results if row.get("error") is None and row.get("usage")]
    outputs = sorted(row["usage"]["output_tokens"] for row in usable)
    inputs = sorted(row["usage"]["input_tokens"] for row in usable)
    truncated = [row for row in usable if row.get("stop_reason") == "length"]

    stop_counts = {}
    for row in usable:
        reason = row.get("stop_reason") or "unknown"
        stop_counts[reason] = stop_counts.get(reason, 0) + 1

    return {
        "model_id": model_id,
        "rows_total": len(results),
        "rows_errored": sum(1 for row in results if row.get("error") is not None),
        "answer_parsed": sum(1 for row in results if row.get("model_answer") is not None),
        "max_new_tokens": cap,
        "input_tokens": {
            "median": _percentile(inputs, 0.5),
            "max": inputs[-1] if inputs else None,
        },
        "output_tokens": {
            "median": _percentile(outputs, 0.5),
            "p95": _percentile(outputs, 0.95),
            "max": outputs[-1] if outputs else None,
        },
        "stop_reason_counts": stop_counts,
        "truncated": len(truncated),
        "truncated_ids": [row["id"] for row in truncated],
        "reasoning_unclosed": sum(1 for row in usable if row.get("reasoning_closed") is False),
    }


def _print_summary(summary, metadata_path=None):
    outputs = summary["output_tokens"]
    inputs = summary["input_tokens"]
    cap = summary["max_new_tokens"]

    print(f"\n=== Run summary: {summary['model_id']} ===")
    print(f"Rows in results file:         {summary['rows_total']}   errored: {summary['rows_errored']}")
    print(f"Answer parsed:                {summary['answer_parsed']} / {summary['rows_total']}")
    print(f"Input tokens  median/max:     {inputs['median']} / {inputs['max']}")
    print(f"Output tokens median/p95/max: {outputs['median']} / {outputs['p95']} / {outputs['max']}")
    counts = "   ".join(f"{k} {v}" for k, v in sorted(summary["stop_reason_counts"].items()))
    print(f"stop_reason:                  {counts or 'n/a'}")
    if summary["reasoning_unclosed"]:
        print(f"Reasoning blocks unclosed:    {summary['reasoning_unclosed']}")

    if summary["truncated"]:
        shown = ", ".join(summary["truncated_ids"][:8])
        more = ", ..." if len(summary["truncated_ids"]) > 8 else ""
        print(
            f"\n[!] {summary['truncated']} response(s) hit max_new_tokens={cap}. Those lengths are a\n"
            f"    LOWER BOUND, not a measurement -- raise the cap in model_configs/ and rerun\n"
            f"    with --overwrite before trusting these scores.\n"
            f"    Truncated: {shown}{more}"
        )
    elif outputs["max"] is not None:
        print(
            f"\n[ok] Nothing hit max_new_tokens={cap}; observed max {outputs['max']}. "
            "This is a real measurement -- document it."
        )
    if metadata_path:
        print(f"Run metadata: {metadata_path}")


def _load_existing_results(result_path, overwrite):
    if overwrite or not os.path.isfile(result_path):
        return []
    with open(result_path, "r", encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list):
        raise ValueError(f"Expected a JSON array in {result_path}")
    return results


def _save_result(result_path, results):
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def execution_pipeline(model_id, year=None, overwrite=False, limit=None):
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        print("Missing runtime dependencies for Hugging Face benchmarking.", file=sys.stderr)
        print(f"Import error: {_IMPORT_ERROR}", file=sys.stderr)
        print("Install torch and transformers in the active Python environment, then rerun the benchmark.", file=sys.stderr)
        return

    print("Starting benchmarking the model via Hugging Face ...\n")
    print(f"Model: {model_id}")
    print(f"Year: {year if year is not None else 'all'}")
    print(f"Limit: {limit if limit is not None else 'all'}")

    model_cfg = _load_model_config(model_id)
    print(f"Model config: {model_cfg}")

    system_prompt = get_system_prompt()
    fingerprint = _config_fingerprint(model_cfg)

    gen_config = dict(model_cfg.get("generation", {}))
    gen_config.setdefault("max_new_tokens", 64)

    temperature = gen_config.get("temperature", 0.0)
    if temperature is None:
        temperature = 0.0
    try:
        temperature = float(temperature)
    except (TypeError, ValueError):
        temperature = 0.0

    if temperature > 0:
        gen_config["do_sample"] = True
    else:
        gen_config["do_sample"] = False

    gen_config["temperature"] = temperature

    questions = _load_questions(year, limit)
    result_path = _result_path(model_id)
    results = _load_existing_results(result_path, overwrite)
    _check_config_drift(results, fingerprint, model_id)

    completed = _completed_ids(results)
    pending = [q for q in questions if (q.get("id"), _RUN_ID) not in completed]

    print(f"Questions selected: {len(questions)}")
    print(f"Questions file: {os.path.basename(_QUESTIONS_PATH)}")
    print(f"Results file: {result_path}")

    # Checked before the model is loaded, never after. Resolving a 7B checkpoint
    # costs minutes and gigabytes, and a fully-processed model needs none of it --
    # that is what lets a notebook be re-run top to bottom without the operator
    # having to track which models are already done.
    if not pending:
        print(
            f"\n[Skip] {model_id}: all {len(questions)} selected questions already "
            "complete. Model not loaded."
        )
        if results:
            _print_summary(_summarize_run(results, model_id, gen_config.get("max_new_tokens")))
        return

    if len(pending) < len(questions):
        print(
            f"Resuming: {len(questions) - len(pending)} already complete, "
            f"{len(pending)} still to run."
        )

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Target Compute Device: {device.upper()}")

    torch.manual_seed(model_cfg.get("seed", 0))

    dtype_str = model_cfg.get("torch_dtype", "float16")
    torch_dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16 if dtype_str == "bfloat16" else "auto"
    trust_remote_code = model_cfg.get("trust_remote_code", False)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if torch_dtype != "auto":
            model_kwargs["dtype"] = torch_dtype
        else:
            model_kwargs["torch_dtype"] = torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        ).to(device)
    except Exception as exc:
        print("\nUnable to load model from Hugging Face.", file=sys.stderr)
        print(f"Model: {model_id}", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        print("Please authenticate with Hugging Face for gated models or pass a public model id.", file=sys.stderr)
        return

    # Written before the first question, not after the last: a run that dies at
    # question 47 is exactly the run whose provenance you need, and a sidecar
    # only written on success would not exist for it.
    run_uid = _run_uid(model_id)
    metadata_path = os.path.join(_RUNS_DIR, run_uid, "metadata.json")
    metadata = {
        "run_uid": run_uid,
        "started_at": _utc_now(),
        "finished_at": None,
        "argv": sys.argv,
        "git": _git_state(),
        "device": device,
        "device_name": _device_name(device),
        "versions": _versions(),
        "model_config": model_cfg,
        "config_sha": fingerprint,
        "questions_file": os.path.basename(_QUESTIONS_PATH),
        "questions_sha256": _sha256_file(_QUESTIONS_PATH),
        "questions_selected": len(questions),
        "questions_pending": len(pending),
        "system_prompt_sha256": _sha256_text(system_prompt),
        "results_file": os.path.basename(result_path),
        "summary": None,
    }
    _write_run_metadata(metadata_path, metadata)
    print(f"Run metadata: {metadata_path}")

    reasoning = model_cfg.get("reasoning") or {}
    declared_open_tag = reasoning.get("open_tag")
    declared_close_tag = reasoning.get("close_tag")

    for question in pending:
        print(f"\n[{question['id']}]")
        try:
            generated = generate_hf_response(
                model,
                tokenizer,
                question["original_question"],
                device,
                system_prompt,
                gen_config,
            )
        except Exception as exc:
            print(f"[Question failed]: {exc}", file=sys.stderr)
            generated = {
                "text": "",
                "input_tokens": None,
                "output_tokens": None,
                "stop_reason": None,
                "error": f"question_failed: {exc}",
            }

        response = generated["text"]

        flagged_tag = detect_undeclared_reasoning_tag(response, declared_open_tag)

        # Strip with the declared pair when the config has one, otherwise with
        # the pair the response itself opened with. Phi-4-mini-reasoning and
        # Magistral declare no tags in any config file and emit them as learned
        # behavior, so the config cannot supply them and the response must.
        open_tag, close_tag = declared_open_tag, declared_close_tag
        if not close_tag and flagged_tag:
            inner = flagged_tag[1:-1]
            open_tag = flagged_tag
            close_tag = f"</{inner}>" if flagged_tag.startswith("<") else f"[/{inner}]"

        parsed = parse_model_response(response, open_tag, close_tag)

        if flagged_tag:
            print(
                f"[Reasoning-tag warning] {question['id']}: response opens with "
                f"{flagged_tag!r}, which isn't this model's declared reasoning tag "
                f"({declared_open_tag!r}). Check whether the config's \"reasoning\" "
                "field needs to be added or corrected.",
                file=sys.stderr,
            )

        result = dict(question)
        result.update(parsed)
        result.update({
            "run_uid": run_uid,
            "run_id": _RUN_ID,
            "model_id": model_id,
            "provider": "huggingface",
            "backend": "local",
            "config_sha": fingerprint,
            "created_at": _utc_now(),
            "usage": {
                "input_tokens": generated["input_tokens"],
                "output_tokens": generated["output_tokens"],
            },
            "stop_reason": generated["stop_reason"],
            # None, not False, when the model declares no reasoning tag: "this
            # model has no thinking block" and "its thinking block was cut off"
            # are different facts and must not collapse into one value.
            "reasoning_closed": (
                declared_close_tag.lower() in response.lower()
                if declared_close_tag else None
            ),
            "error": generated["error"],
        })
        results.append(result)
        _save_result(result_path, results)
        print(
            f"Model answer: {parsed['model_answer']}   "
            f"[{generated['output_tokens']} tok, stop={generated['stop_reason']}]"
        )
        sys.stdout.flush()

    summary = _summarize_run(results, model_id, gen_config.get("max_new_tokens"))
    metadata["finished_at"] = _utc_now()
    metadata["summary"] = summary
    _write_run_metadata(metadata_path, metadata)
    _print_summary(summary, metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the LEET-Arg benchmark against an HF model.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id (dense text decoder models only; not MoE/multimodal). "
             "Examples: meta-llama/Llama-3.2-1B-Instruct (default), "
             "Qwen/Qwen3-4B, microsoft/Phi-4-mini-instruct",
    )
    parser.add_argument("--year", help="Run only questions whose id starts with YEAR_.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N selected questions. Completeness is measured against "
             "the selected set, so --limit 3 then no limit resumes at question 4 rather "
             "than restarting.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear the model result file before writing responses. Needed after changing "
             "a model config, since existing rows are otherwise kept and skipped.",
    )
    args = parser.parse_args()
    try:
        execution_pipeline(args.model, args.year, args.overwrite, args.limit)
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)