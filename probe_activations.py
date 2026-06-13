# File: probe_activations.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_LAYER = 8  # 8 is the Middle layer for Llama 3.2 1B; adapt for other models

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

def extract_layer_activation(model, tokenizer, user_content, layer_idx):
    activation_storage = {}
    
    def hook_fn(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        # Record activation state of the final prompt token
        activation_storage['features'] = hidden_states[0, -1, :].detach().cpu()

    target_layer_module = model.model.layers[layer_idx]
    hook_handle = target_layer_module.register_forward_hook(hook_fn)
    
    # Format the prompt using the proper chat structure
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
        
    hook_handle.remove()
    return activation_storage['features']

def main():
    print("Loading model for contextual activation tracking...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    
    # Replace with prompt pairs from benchmark_repository.py to test for those
    user_present = "Evaluate. Infringing Product: Available. Third-Party Substitute: Available."
    user_absent  = "Evaluate. Infringing Product: Available. Third-Party Substitute: Absent."
    
    print("Extracting representations...")
    act_present = extract_layer_activation(model, tokenizer, user_present, TARGET_LAYER)
    act_absent  = extract_layer_activation(model, tokenizer, user_absent, TARGET_LAYER)
    
    concept_vector = act_present - act_absent
    torch.save(concept_vector, "ip_concept_vector.pt")
    print(f"Success! Concept vector saved. Magnitude: {torch.norm(concept_vector):.4f}")

if __name__ == "__main__":
    main()