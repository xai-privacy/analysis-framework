# File: run_benchmark.py
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import SYSTEM_PROMPT

### Pass the model with --model. Activate one model at a time to avoid stressing
### local resources. Dense text decoder models only -- MoE or multimodal models
### (e.g. Qwen3.5) are not supported. Example values:
###   meta-llama/Llama-3.2-1B-Instruct  (default)
###   Qwen/Qwen3-4B
###   microsoft/Phi-4-mini-instruct

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

def generate_hf_response(model, tokenizer, user_content, device):
    """Generates a text completion natively using the proper chat template sequence."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=300, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_tokens[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

def execution_pipeline(model_id):
    print("Starting benchmarking the model via Hugging Face ...\n")
    print(f"Model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Target Compute Device: {device.upper()}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    for test_case in benchmark_repository:
        print("\n" + "="*70)
        print(f"ID: {test_case['id']}")
        print(f"Target Ground Truth: {test_case['ground_truth']}")
        print("="*70)

        # 1. Execute Prompt A
        print(f"\n[Prompt A]: {test_case['prompt_A']}")
        response_A = generate_hf_response(model, tokenizer, test_case["prompt_A"], device)
        print(f"[Full Response A]:\n{response_A}")

        print("-" * 50)

        # 2. Execute Prompt B
        print(f"[Prompt B]: {test_case['prompt_B']}")
        response_B = generate_hf_response(model, tokenizer, test_case["prompt_B"], device)
        print(f"[Full Response B]:\n{response_B}")

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
    args = parser.parse_args()
    execution_pipeline(args.model)