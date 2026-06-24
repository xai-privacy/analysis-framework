# File: steer_inference.py
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import SYSTEM_PROMPT

TARGET_LAYER = None  # Derived at runtime in main() as the model's middle layer

def get_verdict_ids(tokenizer):
    """Return force_words_ids constraining generation to AWARDED or DENIED."""
    candidates = ["AWARDED", "DENIED", " AWARDED", " DENIED"]
    token_ids = [tokenizer.encode(w, add_special_tokens=False) for w in candidates if tokenizer.encode(w, add_special_tokens=False)]
    return token_ids

def evaluate_with_steering(model, tokenizer, user_query, concept_vector, device, alpha=0.0):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    
    # Hook function to continuously apply steering across the token generation sequence
    def steering_hook(module, args):
        hidden_states = args[0]
        # Instead of targeting just the last token (-1), we apply it across the active sequence window
        if hidden_states.shape[1] == 1:
            # Applies during the auto-regressive generation phase (token by token)
            hidden_states[0, :, :] = hidden_states[0, :, :] + (alpha * concept_vector)
        else:
            # Applies during the initial prompt ingestion phase
            hidden_states[0, -1, :] = hidden_states[0, -1, :] + (alpha * concept_vector)
        return args

    hook_handle = None
    if alpha != 0.0:
        hook_handle = model.model.layers[TARGET_LAYER].register_forward_pre_hook(steering_hook)
        
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            force_words_ids=get_verdict_ids(tokenizer),
        )
        
    if hook_handle is not None:
        hook_handle.remove()
        
    return tokenizer.decode(output_tokens[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

def main(model_id):
    print(f"Model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise SystemExit(
            f"{model_id} does not expose model.model.layers; this pipeline supports "
            "dense text decoder models only (not MoE/multimodal)."
        )

    # Derive the middle layer at runtime so this adapts to any model depth.
    # evaluate_with_steering() reads TARGET_LAYER as a module global, so reassign it here.
    global TARGET_LAYER
    TARGET_LAYER = len(model.model.layers) // 2
    print(f"Model has {len(model.model.layers)} layers; injecting steering at middle layer {TARGET_LAYER}.")
    print("NOTE: probe and steer must be run with the same --model. They share "
          "ip_concept_vector.pt, and a model/layer mismatch silently produces meaningless results.")

    try:
        concept_vector = torch.load("ip_concept_vector.pt").to(device).to(torch.float16)
    except FileNotFoundError:
        print("Error: ip_concept_vector.pt not found. Run probe_activations.py first.")
        return

    # Replace with prompt pairs from benchmark_repository.py to test for those
    prompts = {
        "Prompt A (Z=1, Should be DENIED)": "Evaluate. Infringing Product: Available. Third-Party Substitute: Available.",
        "Prompt B (Z=0, Should be AWARDED)": "Evaluate. Infringing Product: Available. Third-Party Substitute: Not Available."
    }
    
    print("\n================== PHASE 1: UN-STEERED BASELINE ==================")
    for name, query in prompts.items():
        print(f"\n[{name}]:")
        response = evaluate_with_steering(model, tokenizer, query, concept_vector, device, alpha=0.0)
        print(response)
        print("-" * 50)
        
    print("\n================== PHASE 2: STEERED INTERVENTION ==================")
    
    # Prompt A does not require correction, so we run it with alpha = 0.0
    print(f"\n[Prompt A - Unsteered Baseline (α = 0.0)]:")
    response_A = evaluate_with_steering(model, tokenizer, prompts["Prompt A (Z=1, Should be DENIED)"], concept_vector, device, alpha=0.0)
    print(response_A)
    print("-" * 50)
    
    # Prompt B is broken, so we apply negative steering to suppress Z presence paths and flip it to AWARDED
    alpha_value = -3.8
    print(f"[Prompt B - Steered Intervention (α = {alpha_value})]:")
    print(f"> Applying negative steering to force model to prioritize Z=0 behavior...")
    response_B = evaluate_with_steering(model, tokenizer, prompts["Prompt B (Z=0, Should be AWARDED)"], concept_vector, device, alpha=alpha_value)
    print(response_B)
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject a concept vector at a model's middle layer.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HF model id (dense text decoder models only; not MoE/multimodal). "
             "Examples: meta-llama/Llama-3.2-1B-Instruct (default), "
             "Qwen/Qwen3-4B, microsoft/Phi-4-mini-instruct. "
             "Must match the --model used for probe_activations.py.",
    )
    args = parser.parse_args()
    main(args.model)