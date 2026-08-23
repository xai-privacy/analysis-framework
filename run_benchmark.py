# File: run_benchmark.py
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
    with open(fallback_path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_model_response(response):
    """Extract the answer marker and retain the remaining response as rationale."""
    answer_match = re.search(
        r"\bAnswer\s*-\s*\{?\s*([0-9]+|[A-Ea-e]|[①②③④⑤])"
        r"(?=\s*(?:\}|[.:;,)]|$))\s*\}?\s*[.:]?\s*",
        response,
        re.IGNORECASE,
    )
    if answer_match is None:
        return {"model_answer": None, "model_rationale": response.strip()}

    rationale = (response[:answer_match.start()] + response[answer_match.end():]).strip()
    return {"model_answer": answer_match.group(1).strip(), "model_rationale": rationale}


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