import argparse
import json
import os
import re
import sys
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
    "LEET_Arg_Questions_cleaned.json",
)
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slm_results")
_FALLBACK_CONFIG_NAME = "meta-llama_Llama-3.2-1B-Instruct.json"


def _fetch_config_from_hf(model_id):
    """Attempt to build a model config from the HF Hub's config.json /
    generation_config.json. Decoding is forced to greedy (temperature=0,
    do_sample=False) regardless of what the repo specifies. Also inspects
    the model's chat template -- checking both tokenizer_config.json's
    embedded "chat_template" field and a standalone chat_template.jinja
    file, since repos use either convention -- for a <think>...</think>
    pattern, and if found, populates a "reasoning" block so
    detect_undeclared_reasoning_tag() doesn't flag it as unexpected. Returns
    None on any failure (missing files, network error, huggingface_hub
    unavailable) so the caller can fall back."""
    if hf_hub_download is None:
        print(
            "[Config warning] huggingface_hub is not available; cannot auto-fetch config.",
            file=sys.stderr,
        )
        return None

    cfg = {
        "model_id": model_id,
        "torch_dtype": "float16",
        "trust_remote_code": False,
        "seed": 0,
        "generation": {"do_sample": False, "temperature": 0.0, "max_new_tokens": 1024},
    }

    fetched_anything = False

    try:
        path = hf_hub_download(model_id, "config.json")
        with open(path, "r", encoding="utf-8") as f:
            model_cfg = json.load(f)
        fetched_anything = True

        dtype = model_cfg.get("torch_dtype")
        if dtype in ("float16", "bfloat16"):
            cfg["torch_dtype"] = dtype
        elif dtype is not None:
            cfg["torch_dtype"] = "auto"

        if "auto_map" in model_cfg:
            cfg["trust_remote_code"] = True
    except EntryNotFoundError:
        pass
    except Exception as exc:
        print(f"[Config warning] Failed to fetch config.json for {model_id}: {exc}", file=sys.stderr)
        return None

    try:
        path = hf_hub_download(model_id, "generation_config.json")
        with open(path, "r", encoding="utf-8") as f:
            gen_cfg = json.load(f)
        fetched_anything = True

        # Only length controls are taken from the repo; sampling behavior
        # is forced below regardless of what this file specifies.
        if "max_new_tokens" in gen_cfg:
            cfg["generation"]["max_new_tokens"] = gen_cfg["max_new_tokens"]
    except EntryNotFoundError:
        pass
    except Exception as exc:
        print(
            f"[Config warning] Failed to fetch generation_config.json for {model_id}: {exc}",
            file=sys.stderr,
        )
        return None

    # Forced, non-negotiable: greedy decoding for auto-fetched configs.
    cfg["generation"]["do_sample"] = False
    cfg["generation"]["temperature"] = 0.0

    # Heuristic: if the model's chat template references <think>...</think>,
    # it's likely a "thinking" model that wraps reasoning in that tag. This
    # only *adds* a reasoning block when detected; absence of a match means
    # no "reasoning" key is set, same as before.
    #
    # The template can live in either of two places depending on the repo:
    #   - embedded in tokenizer_config.json's "chat_template" field, or
    #   - a standalone chat_template.jinja file (e.g. LiquidAI/LFM2.5-*),
    #     which newer `transformers`/Hub conventions increasingly use instead
    #     of embedding it inline.
    # Check both; a hit on either is enough.
    chat_template = ""

    try:
        path = hf_hub_download(model_id, "tokenizer_config.json")
        with open(path, "r", encoding="utf-8") as f:
            tokenizer_cfg = json.load(f)
        fetched_anything = True

        template_field = tokenizer_cfg.get("chat_template", "")
        if isinstance(template_field, list):
            # Some repos store multiple named templates as a list of dicts.
            template_field = " ".join(
                str(entry.get("template", "")) for entry in template_field if isinstance(entry, dict)
            )
        chat_template += str(template_field)
    except EntryNotFoundError:
        pass
    except Exception as exc:
        print(
            f"[Config warning] Failed to inspect tokenizer_config.json for {model_id}: {exc}",
            file=sys.stderr,
        )
        # Non-fatal: reasoning-tag detection is best-effort, doesn't block the fetch.

    try:
        path = hf_hub_download(model_id, "chat_template.jinja")
        with open(path, "r", encoding="utf-8") as f:
            chat_template += f.read()
        fetched_anything = True
    except EntryNotFoundError:
        pass
    except Exception as exc:
        print(
            f"[Config warning] Failed to fetch chat_template.jinja for {model_id}: {exc}",
            file=sys.stderr,
        )
        # Non-fatal, same reasoning as above.

    if "<think>" in chat_template and "</think>" in chat_template:
        cfg["reasoning"] = {"open_tag": "<think>", "close_tag": "</think>"}

    if not fetched_anything:
        print(
            f"[Config warning] No config.json or generation_config.json found on the Hub for {model_id}.",
            file=sys.stderr,
        )
        return None

    return cfg


def _load_model_config(model_id):
    """Resolve a model config with a three-tier strategy:
    1. Local model_configs/<sanitized_model_id>.json, if present.
    2. Auto-fetched from the HF Hub (config.json / generation_config.json),
       with decoding forced to greedy (do_sample=False, temperature=0).
       A successful fetch is cached to disk so later runs skip the fetch.
    3. Fall back to the Llama default config, with an explicit warning.
    """
    sanitized = model_id.replace("/", "_")
    config_path = os.path.join(_MODEL_CONFIGS_DIR, f"{sanitized}.json")

    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(
        f"[Config warning] No config found at {config_path} -- attempting to fetch "
        f"{model_id}'s config from the Hugging Face Hub...",
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
                f"[Config] Auto-fetched config for {model_id} saved to {config_path} "
                "(do_sample=False, temperature=0.0 forced).",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[Config warning] Fetched config but failed to cache it: {exc}", file=sys.stderr)
        return cfg

    # Fall back to Llama config as default
    fallback_path = os.path.join(_MODEL_CONFIGS_DIR, _FALLBACK_CONFIG_NAME)
    print(
        f"[Config warning] Could not fetch a config for {model_id} from the Hub -- falling back to "
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


def generate_hf_response(model, tokenizer, user_content, device, system_prompt, gen_config):
    """Generates a text completion natively using the proper chat template sequence."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

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
        return json.dumps({"error": "generation_failed", "reason": str(exc)})

    return tokenizer.decode(
        output_tokens[0][prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


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


def execution_pipeline(model_id, year=None, overwrite=False):
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        print("Missing runtime dependencies for Hugging Face benchmarking.", file=sys.stderr)
        print(f"Import error: {_IMPORT_ERROR}", file=sys.stderr)
        print("Install torch and transformers in the active Python environment, then rerun the benchmark.", file=sys.stderr)
        return

    print("Starting benchmarking the model via Hugging Face ...\n")
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

    questions = _load_questions(year)
    result_path = _result_path(model_id)
    results = _load_existing_results(result_path, overwrite)

    print(f"Questions selected: {len(questions)}")
    print(f"Results file: {result_path}")
    for question in questions:
        print(f"\n[{question['id']}]")
        try:
            response = generate_hf_response(
                model,
                tokenizer,
                question["original_question"],
                device,
                system_prompt,
                gen_config,
            )
        except Exception as exc:
            print(f"[Question failed]: {exc}", file=sys.stderr)
            response = f"Generation failed: {exc}"

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
        results.append(result)
        _save_result(result_path, results)
        print(f"Model answer: {parsed['model_answer']}")
        sys.stdout.flush()

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
    args = parser.parse_args()
    try:
        execution_pipeline(args.model, args.year, args.overwrite)
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)