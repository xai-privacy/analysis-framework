# File: run_benchmark.py
import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

import run_manifest
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

_MODEL_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_configs")
_QUESTIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "benchmarks",
    "LEET_Arg_Questions_cleaned.json",
)
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slm_results")


def _load_model_config(model_id):
    """Load model-specific config from model_configs/<sanitized_model_id>.json.
    Returns a dict with keys: torch_dtype, trust_remote_code, seed, generation.
    Falls back to the Llama config if no model-specific config file exists."""
    sanitized = model_id.replace("/", "_")
    config_path = os.path.join(_MODEL_CONFIGS_DIR, f"{sanitized}.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Fall back to Llama config as default
    fallback_path = os.path.join(_MODEL_CONFIGS_DIR, "meta-llama_Llama-3.2-1B-Instruct.json")
    print(
        f"[Config warning] No config found at {config_path} -- falling back to "
        "meta-llama/Llama-3.2-1B-Instruct's settings (non-thinking, "
        "do_sample=False, max_new_tokens=1024). Add a model_configs/"
        f"{sanitized}.json with this model's actual recommended settings.",
        file=sys.stderr,
    )
    with open(fallback_path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_model_response(response):
    """Extract the answer marker and retain the remaining response as rationale."""
    search_text = response
    offset = 0
    if "<think>" in response.lower():
        close_idx = response.lower().find("</think>")
        if close_idx == -1:
            # Reasoning never finished, so there is no reliable final answer.
            return {"model_answer": None, "model_rationale": response.strip()}
        offset = close_idx + len("</think>")
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


def _eos_token_ids(model, tokenizer):
    """Every token id that legitimately terminates generation.

    Both sources are needed: chat models such as Llama-3.2 declare a list on
    generation_config, and the token that actually fires (<|eot_id|>) is not the
    tokenizer's eos_token_id. Checking only the tokenizer mislabels almost every
    response as an unrecognised stop.
    """
    ids = set()
    sources = (
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
    )
    for source in sources:
        if source is None:
            continue
        if isinstance(source, (list, tuple, set)):
            ids.update(int(i) for i in source if i is not None)
        else:
            ids.add(int(source))
    return ids


def generate_hf_response_verbose(model, tokenizer, user_content, device, system_prompt, gen_config):
    """Same generation as generate_hf_response, plus per-call telemetry.

    Returns a dict of text, prompt_tokens, completion_tokens, stop_reason,
    gen_seconds and max_new_tokens. return_dict_in_generate only changes the
    return container, never the tokens that get sampled.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # add_special_tokens=False because the chat template already emitted BOS.
    # Re-tokenizing without this prepends a second <|begin_of_text|>, which for
    # Llama-3.2 collapses the response to a bare "Answer-N" with no rationale --
    # the models look tersely uncooperative rather than misconfigured.
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = int(inputs["input_ids"].shape[1])

    generation_kwargs = dict(gen_config)
    max_new_tokens = generation_kwargs.get("max_new_tokens")
    generation_kwargs["return_dict_in_generate"] = True

    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs
            )
    except Exception as exc:
        print(f"[Generation Error]: {exc}", file=sys.stderr)
        return {
            "text": json.dumps({"error": "generation_failed", "reason": str(exc)}),
            "prompt_tokens": prompt_len,
            "completion_tokens": 0,
            "stop_reason": "error",
            "gen_seconds": round(time.perf_counter() - started, 3),
            "max_new_tokens": max_new_tokens,
        }

    if device == "cuda":
        torch.cuda.synchronize()
    gen_seconds = time.perf_counter() - started

    sequences = getattr(outputs, "sequences", outputs)
    generated = sequences[0][prompt_len:]
    completion_tokens = int(generated.shape[0])

    # Order matters: generate() appends EOS and then stops, so a response ending
    # on EOS at exactly the cap satisfies both tests. Check the token first.
    if completion_tokens and int(generated[-1]) in _eos_token_ids(model, tokenizer):
        stop_reason = "eos"
    elif max_new_tokens is not None and completion_tokens >= int(max_new_tokens):
        stop_reason = "length"
    else:
        stop_reason = "stop"

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return {
        "text": text,
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_tokens,
        "stop_reason": stop_reason,
        "gen_seconds": round(gen_seconds, 3),
        "max_new_tokens": max_new_tokens,
    }


def generate_hf_response(model, tokenizer, user_content, device, system_prompt, gen_config):
    """Generates a text completion natively using the proper chat template sequence."""
    return generate_hf_response_verbose(
        model, tokenizer, user_content, device, system_prompt, gen_config
    )["text"]


def _result_path(model_id):
    signature = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
    return os.path.join(_RESULTS_DIR, f"{signature}.json")


def _load_questions(year=None):
    with open(_QUESTIONS_PATH, "r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if year is None:
        return questions
    year_prefix = f"{year}_"
    return [question for question in questions if str(question.get("id", "")).startswith(year_prefix)]


def _select_questions(questions, limit=None, ids=None):
    """Subset the question list.

    --ids is an explicit ordered list; --limit takes a round-robin sample across
    year buckets so a small subset spans the dataset instead of being all 2021.
    """
    if ids:
        wanted = [qid.strip() for qid in ids.split(",") if qid.strip()]
        by_id = {question["id"]: question for question in questions}
        missing = [qid for qid in wanted if qid not in by_id]
        if missing:
            raise ValueError(f"Unknown question ids: {', '.join(missing)}")
        return [by_id[qid] for qid in wanted]

    if limit is None or limit >= len(questions):
        return questions

    buckets = OrderedDict()
    for question in questions:
        year = str(question.get("year") or str(question["id"]).split("_")[0])
        buckets.setdefault(year, []).append(question)

    chosen = set()
    depth = 0
    while len(chosen) < limit:
        progressed = False
        for bucket in buckets.values():
            if depth < len(bucket):
                chosen.add(bucket[depth]["id"])
                progressed = True
                if len(chosen) == limit:
                    break
        if not progressed:
            break
        depth += 1

    return [question for question in questions if question["id"] in chosen]


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


def execution_pipeline(model_id, year=None, overwrite=False, limit=None, ids=None,
                       max_new_tokens=None, resume=False, revision=None, cli_args=None):
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        print("Missing runtime dependencies for Hugging Face benchmarking.", file=sys.stderr)
        print(f"Import error: {_IMPORT_ERROR}", file=sys.stderr)
        print("Install torch and transformers in the active Python environment, then rerun the benchmark.", file=sys.stderr)
        return

    run_id = run_manifest.new_run_id(model_id)
    _, restore_console = run_manifest.open_console_log(run_id)
    try:
        _execute_run(model_id, year, overwrite, limit, ids, max_new_tokens,
                     resume, revision, cli_args, run_id)
    finally:
        restore_console()


def _execute_run(model_id, year, overwrite, limit, ids, max_new_tokens,
                 resume, revision, cli_args, run_id):
    """The run itself. Split out so console teeing wraps every exit path."""
    print("Starting benchmarking the model via Hugging Face ...\n")
    print(f"Run id: {run_id}")
    print(f"Model: {model_id}")
    print(f"Year: {year if year is not None else 'all'}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Target Compute Device: {device.upper()}")

    model_cfg = _load_model_config(model_id)
    print(f"Model config: {model_cfg}")

    system_prompt = get_system_prompt()

    torch.manual_seed(model_cfg.get("seed", 0))

    dtype_str = model_cfg.get("torch_dtype", "float16")
    torch_dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16 if dtype_str == "bfloat16" else "auto"
    trust_remote_code = model_cfg.get("trust_remote_code", False)

    gen_config = dict(model_cfg.get("generation", {}))
    gen_config.setdefault("max_new_tokens", 64)

    if max_new_tokens is not None:
        print(
            f"[Override] max_new_tokens={max_new_tokens} "
            f"(config said {gen_config['max_new_tokens']})"
        )
        gen_config["max_new_tokens"] = int(max_new_tokens)

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

    try:
        load_kwargs = {"revision": revision} if revision else {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        model_kwargs = {"trust_remote_code": trust_remote_code, **load_kwargs}
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

    if ids and year:
        print("[Note] --ids given; ignoring --year.", file=sys.stderr)
    questions = _select_questions(_load_questions(None if ids else year), limit=limit, ids=ids)

    result_path = _result_path(model_id)
    results = _load_existing_results(result_path, overwrite)

    if resume:
        done_ids = {record.get("id") for record in results}
        before = len(questions)
        questions = [question for question in questions if question["id"] not in done_ids]
        print(f"[Resume] skipping {before - len(questions)} of {before} ids already in {result_path}")

    manifest = run_manifest.build_manifest(
        run_id=run_id,
        model_id=model_id,
        model_cfg=model_cfg,
        gen_config=gen_config,
        system_prompt=system_prompt,
        questions=questions,
        device=device,
        torch_module=torch,
        questions_path=_QUESTIONS_PATH,
        cli_args=cli_args,
        revision=revision,
        tokenizer=tokenizer,
    )
    manifest_path = run_manifest.write_manifest(run_id, manifest)

    print(f"Questions selected: {len(questions)}")
    print(f"Results file: {result_path}")
    print(f"Run manifest: {manifest_path}")

    results_for_run = []
    try:
        for question in questions:
            print(f"\n[{question['id']}]")
            try:
                telemetry = generate_hf_response_verbose(
                    model,
                    tokenizer,
                    question["original_question"],
                    device,
                    system_prompt,
                    gen_config,
                )
            except Exception as exc:
                print(f"[Question failed]: {exc}", file=sys.stderr)
                telemetry = {
                    "text": f"Generation failed: {exc}",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "stop_reason": "error",
                    "gen_seconds": None,
                    "max_new_tokens": gen_config.get("max_new_tokens"),
                }
            response = telemetry["text"]

            parsed = parse_model_response(response)

            declared_open_tag = (model_cfg.get("reasoning") or {}).get("open_tag")
            flagged_tag = detect_undeclared_reasoning_tag(response, declared_open_tag)
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
            result["raw_response"] = response
            result["run_id"] = run_id
            for key in ("prompt_tokens", "completion_tokens", "stop_reason",
                        "gen_seconds", "max_new_tokens"):
                result[key] = telemetry[key]

            results.append(result)
            results_for_run.append(result)
            _save_result(result_path, results)
            print(
                f"Model answer: {parsed['model_answer']}  "
                f"[{telemetry['completion_tokens']} tok / {telemetry['stop_reason']} / "
                f"{telemetry['gen_seconds']}s]"
            )
            sys.stdout.flush()
    finally:
        # Runs get killed by Colab timeouts; a manifest for the questions that
        # did finish is more useful than none.
        run_manifest.finalize(run_id, manifest, results_for_run)
        run_manifest.append_index(manifest)

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
        "--overwrite",
        action="store_true",
        help="Clear the model result file before writing responses.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only N questions, sampled round-robin across years.",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated question ids, e.g. 2021_02,2024_05. Overrides --year and --limit.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        dest="max_new_tokens",
        help="Override the model config's generation.max_new_tokens for this run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip ids already present in the result file instead of appending duplicates.",
    )
    parser.add_argument(
        "--revision",
        help="Pin the HF model revision (branch, tag, or commit SHA). "
             "Recorded in the run manifest either way.",
    )
    args = parser.parse_args()
    try:
        execution_pipeline(
            args.model,
            args.year,
            args.overwrite,
            limit=args.limit,
            ids=args.ids,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
            revision=args.revision,
            cli_args=vars(args),
        )
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)