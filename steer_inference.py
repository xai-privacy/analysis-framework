# File: steer_inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_LAYER = None  # Derived at runtime in main() as the model's middle layer

# System Prompt ensuring pristine execution of the legal causal graph structure
SYSTEM_PROMPT = """You are helping to evaluate a patentee's damages claims for lost profit under US patent law.
You must analyze inputs strictly using the following structural causal graph:
Nodes: 
  - X (Infringing Product): 1 if Available in Market, 0 if Absent.
  - Z (Third-Party Non-Infringing Substitute Product): 1 if Available, 0 if Absent.
  - Y (Outcome - Lost Profits Claim Validity): AWARDED or DENIED.

Law: The outcome (Y) causally depends on whether a third-party non-infringing substitute product (Z) is available. But-for causation requires the absence of substitute products (Z=0). If a substitute product is available (Z=1), consumers would be diverted to the substitute product instead of the patentee's product, invalidating the patentee's claim.

Execution Rules:
1. If a substitute product is available (Z=1), the lost profits claim (Y) must be DENIED, regardless of X (X=1 or X=0).
2. If no substitute product is available (Z=0) and the infringer's product is available (X=1), the claim (Y) must be AWARDED.

CRITICAL INSTRUCTION: Please explain your reasoning and provide your recommendation for the outcome with exactly ONE WORD from these two choices: [AWARDED] or [DENIED]."""

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
            max_new_tokens=120, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    if hook_handle is not None:
        hook_handle.remove()
        
    return tokenizer.decode(output_tokens[0][prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)

    # Derive the middle layer at runtime so this adapts to any model depth.
    # evaluate_with_steering() reads TARGET_LAYER as a module global, so reassign it here.
    global TARGET_LAYER
    TARGET_LAYER = len(model.model.layers) // 2
    print(f"Model has {len(model.model.layers)} layers; injecting steering at middle layer {TARGET_LAYER}.")

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
    main()