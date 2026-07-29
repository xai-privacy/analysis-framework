# File: run_benchmark.py
import argparse
import json
import os
import sys

import torch

from prompts import get_system_prompt
from structured_outputs import DecisionOutput, parse_and_reason

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
    from transformers import LogitsProcessorList
except Exception:
    LogitsProcessorList = None

try:
    from transformers import RegexLogitsProcessor
except Exception:
    RegexLogitsProcessor = None

_MODEL_CONFIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_configs")


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

### Usage:
###   python3 run_benchmark.py [--model <hf-model-id>] [--dsl <plain|odrl|legalruleml|de_jure>]
###
### --model : Hugging Face model id. Dense text decoder models only --
###           MoE or multimodal models (e.g. Qwen3.5) are not supported.
###           Examples:
###             meta-llama/Llama-3.2-1B-Instruct  (default)
###             Qwen/Qwen3.5-4B
###             microsoft/Phi-4-mini-instruct
###
### --dsl   : Domain-specific language for the formal rules embedded in the
###           system prompt. Choices:
###             plain        (default) -- plain English rules (no external file)
###             odrl                   -- ODRL policy from odrl_rules.json
###             legalruleml            -- LegalRuleML XML from legal_rules.xml
###             de_jure                -- De Jure structured rules from de_jure_rules.json

benchmark_repository = [
    {
        "id": "IP_Causation_Pair_1",
        "ground_truth": "A: AWARDED; B: DENIED",
        "prompt_A": "Evaluate. Infringing Product: Available. Third-Party Substitute: Not Available.",
        "prompt_B": "Evaluate. Infringing Product: Available. Third-Party Substitute: Available."
    },
    {
        "id": "IP_Causation_Pair_2",
        "ground_truth": "A: DENIED; B: DENIED",
        "prompt_A": "Evaluate. Infringing Product: Available. Third-Party Substitute: Available.",
        "prompt_B": "Evaluate. Infringing Product: Not Available. Third-Party Substitute: Available."
    },
    {
        "id": "IP_Causation_Pair_3",
        "ground_truth": "A: DENIED; B: AWARDED",
        "prompt_A": "Evaluate. Infringing Product: Available. Third-Party Substitute: Available.",
        "prompt_B": "Evaluate. Infringing Product: Available. Third-Party Substitute: Not Available."
    },
    {
        "id": "IP_Causation_Pair_4",
        "ground_truth": "A: DENIED; B: DENIED",
        "prompt_A": "Evaluate. X=1, Z=1.",
        "prompt_B": "Evaluate. X=0, Z=1."
    },
    {
        "id": "IP_Causation_Pair_5",
        "ground_truth": "A: DENIED; B: AWARDED",
        "prompt_A": "Evaluate. X=1, Z=1.",
        "prompt_B": "Evaluate. X=1, Z=0."
    },
    {
        "id": "IP_Causation_Pair_6",
        "ground_truth": "A: AWARDED; B: DENIED",
        "prompt_A": "Evaluate. X=1, Z=0.",
        "prompt_B": "Evaluate. X=1, Z=1."
    }
]

def _build_predicate_constraint(tokenizer):
    """Return a logits processor that constrains output to a JSON object with predicate fields."""
    if RegexLogitsProcessor is None or LogitsProcessorList is None:
        return None
    pattern = (
        r'\{\s*"infringing_product_available"\s*:\s*(true|false)\s*,'
        r'\s*"substitute_product_available"\s*:\s*(true|false)\s*\}'
    )
    try:
        return RegexLogitsProcessor(regex=pattern, tokenizer=tokenizer)
    except TypeError:
        return None


def generate_hf_response(model, tokenizer, user_content, device, system_prompt, gen_config):
    """Generates a text completion natively using the proper chat template sequence."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    logits_processor = None
    try:
        logits_processor = _build_predicate_constraint(tokenizer)
    except Exception:
        logits_processor = None

    generation_kwargs = dict(gen_config)
    if logits_processor is not None and LogitsProcessorList is not None:
        generation_kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

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

    return tokenizer.decode(output_tokens[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def run_structured_decision(model, tokenizer, user_content, device, system_prompt, gen_config):
    """Generate a model response and convert it into a structured decision via the DSL layer."""
    try:
        raw_response = generate_hf_response(model, tokenizer, user_content, device, system_prompt, gen_config)
    except Exception as exc:
        print(f"[Prompt Execution Error]: {exc}", file=sys.stderr)
        raw_response = json.dumps({"error": "generation_failed", "reason": str(exc)})

    try:
        decision = parse_and_reason(raw_response)
    except Exception as exc:
        print(f"[Structured Decision Error]: {exc}", file=sys.stderr)
        decision = DecisionOutput(decision="DENIED", explanation=f"Parsing failed: {exc}")
    return decision, raw_response

def execution_pipeline(model_id, dsl):
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        print("Missing runtime dependencies for Hugging Face benchmarking.", file=sys.stderr)
        print(f"Import error: {_IMPORT_ERROR}", file=sys.stderr)
        print("Install torch and transformers in the active Python environment, then rerun the benchmark.", file=sys.stderr)
        return

    print("Starting benchmarking the model via Hugging Face ...\n")
    print(f"Model: {model_id}")
    print(f"DSL: {dsl}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Target Compute Device: {device.upper()}")

    model_cfg = _load_model_config(model_id)
    print(f"Model config: {model_cfg}")

    system_prompt = get_system_prompt(dsl)

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

    for test_case in benchmark_repository:
        print("\n" + "="*70)
        print(f"ID: {test_case['id']}")
        print(f"Target Ground Truth: {test_case['ground_truth']}")
        print("="*70)

        # 1. Execute Prompt A
        print(f"\n[Prompt A]: {test_case['prompt_A']}")
        try:
            decision_A, response_A = run_structured_decision(model, tokenizer, test_case["prompt_A"], device, system_prompt, gen_config)
        except Exception as exc:
            print(f"[Prompt A failed]: {exc}", file=sys.stderr)
            decision_A = DecisionOutput(decision="DENIED", explanation=f"Prompt execution failed: {exc}")
            response_A = json.dumps({"error": "prompt_failed", "reason": str(exc)})
        print(f"[Final Output (DSL Runtime) A]: {decision_A.model_dump()}")
        print("[Model Output A]:")
        print(response_A)
        sys.stdout.flush()

        print("-" * 50)

        # 2. Execute Prompt B
        print(f"[Prompt B]: {test_case['prompt_B']}")
        try:
            decision_B, response_B = run_structured_decision(model, tokenizer, test_case["prompt_B"], device, system_prompt, gen_config)
        except Exception as exc:
            print(f"[Prompt B failed]: {exc}", file=sys.stderr)
            decision_B = DecisionOutput(decision="DENIED", explanation=f"Prompt execution failed: {exc}")
            response_B = json.dumps({"error": "prompt_failed", "reason": str(exc)})
        print(f"[Final Output (DSL Runtime) B]: {decision_B.model_dump()}")
        print("[Model Output B]:")
        print(response_B)
        sys.stdout.flush()

        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the IP causal-reasoning benchmark against an HF model.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id (dense text decoder models only; not MoE/multimodal). "
             "Examples: meta-llama/Llama-3.2-1B-Instruct (default), "
             "Qwen/Qwen3-4B, microsoft/Phi-4-mini-instruct",
    )
    parser.add_argument(
        "--dsl",
        default="plain",
        choices=["plain", "odrl", "legalruleml", "de_jure"],
        help="Domain-specific language for the formal rules passed to the model. "
             "Choices: plain (default), odrl, legalruleml, or de_jure.",
    )
    args = parser.parse_args()
    try:
        execution_pipeline(args.model, args.dsl)
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)