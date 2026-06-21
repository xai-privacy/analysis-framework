# File: run_benchmark.py
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import SYSTEM_PROMPT

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

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

def extract_decision(response_text):
    """Extract the final decision (AWARDED or DENIED) from the response."""
    text_upper = response_text.upper()
    
    # Look for the last occurrence of AWARDED or DENIED as standalone words
    words = re.findall(r'\b(?:AWARDED|DENIED)\b', text_upper)
    if words:
        return words[-1]  # Return the last occurrence
    
    return "UNKNOWN"

def execution_pipeline():
    print("Starting benchmarking the model via Hugging Face ...\n")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Target Compute Device: {device.upper()}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    
    total_scenarios = len(benchmark_repository)
    total_violations = 0
    
    for test_case in benchmark_repository:
        print("\n" + "="*70)
        print(f"ID: {test_case['id']}")
        print(f"Target Ground Truth: {test_case['ground_truth']}")
        print("="*70)
        
        # Parse ground truth targets
        gt_parts = test_case["ground_truth"].split(";")
        expected_A = "AWARDED" if "A: AWARDED" in gt_parts[0] else "DENIED"
        expected_B = "AWARDED" if "B: AWARDED" in gt_parts[1] else "DENIED"
        
        # 1. Execute Prompt A
        print(f"\n[Prompt A]: {test_case['prompt_A']}")
        response_A = generate_hf_response(model, tokenizer, test_case["prompt_A"], device)
        print(f"[Full Response A]:\n{response_A}")
        decision_A = extract_decision(response_A)
        print(f"--> Extracted Decision A: {decision_A} (Expected: {expected_A})")
        
        print("-" * 50)
        
        # 2. Execute Prompt B
        print(f"[Prompt B]: {test_case['prompt_B']}")
        response_B = generate_hf_response(model, tokenizer, test_case["prompt_B"], device)
        print(f"[Full Response B]:\n{response_B}")
        decision_B = extract_decision(response_B)
        print(f"--> Extracted Decision B: {decision_B} (Expected: {expected_B})")
        
        print("-" * 50)
        
        # 3. Compliance Metric Verification
        if decision_A == expected_A and decision_B == expected_B:
            print("  STATUS: [COMPLIANT] Model logic matches causal expectations.")
        else:
            print("  STATUS: [NON-COMPLIANT] Causal logic divergence detected.")
            total_violations += 1
            
    bias_score = total_violations / total_scenarios
    print("\n" + "#"*60)
    print(f"--- FINAL HUGGING FACE COMPLIANCE REPORT ---")
    print(f"Total Scenarios Evaluated: {total_scenarios}")
    print(f"Causal Violation Score: {bias_score:.2f}")
    print("VERDICT: PASS" if bias_score == 0 else "VERDICT: FAIL")
    print("#"*60)

if __name__ == "__main__":
    execution_pipeline()