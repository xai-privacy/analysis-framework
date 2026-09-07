import argparse
import json
import re
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Parse Command Line Arguments
parser = argparse.ArgumentParser(description="Run argument evaluation over a JSON file with a Hugging Face model.")
parser.add_argument(
    "--model", 
    type=str, 
    default="meta-llama/Llama-3.2-1B-Instruct", 
    help="The Hugging Face model ID"
)
parser.add_argument(
    "--input", 
    type=str, 
    default="results.json", 
    help="Path to the input JSON file containing the extracted claims and statements"
)
args = parser.parse_args()
MODEL_ID = args.model
INPUT_FILE = args.input

# 2. Simplified Pydantic schema for evaluation output
class StatementEvaluation(BaseModel):
    is_correct: bool = Field(description="True if the statement's logical analysis is correct, False if it is incorrect")
    reasoning: str = Field(description="Concise logical explanation of why the statement is correct or incorrect")

# 3. Initialize Hugging Face Model
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model {MODEL_ID} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map=device
)

def evaluate_verbatim_statement(statement_text: str, claims: dict, attacks: list) -> StatementEvaluation:
    system_prompt = f"""You are a formal logic evaluator analyzing a debate context.
Respond ONLY with a valid JSON object. Do not include any markdown formatting, conversational text, or explanations outside the JSON.
Your JSON must strictly adhere to this schema:
{json.dumps(StatementEvaluation.model_json_schema(), indent=2)}
"""

    user_prompt = f"""
    Argument Graph Claims:
    {json.dumps(claims, indent=2)}
    
    Attack Relations (Attacker -> Target):
    {json.dumps(attacks, indent=2)}
    
    Verbatim Evaluation Statement to Analyze:
    "{statement_text}"
    
    Task:
    Evaluate if the provided statement is logically correct based on the argument graph.
    1. Set `is_correct` to True if the statement's assessment is logically accurate, or False if it is logically flawed or incorrect.
    2. Provide your `reasoning` for why the statement is correct or incorrect.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Format prompt and generate
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt",
        return_dict=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Extract response
    prompt_length = inputs["input_ids"].shape[1]
    response_text = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()
    
    # Isolate JSON in case the model hallucinates markdown tags
    json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(1)
        
    # Safely parse JSON to prevent crashes on single statement failures
    try:
        return StatementEvaluation.model_validate_json(response_text)
    except Exception as e:
        return StatementEvaluation(
            is_correct=False, 
            reasoning=f"Failed to parse model output. Raw output: {response_text}"
        )

# 4. Run evaluation loop over the JSON file
print(f"\n--- Loading {INPUT_FILE} ---\n")

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find the file '{INPUT_FILE}'. Please ensure the path is correct.")
    exit(1)

for item in questions_data:
    q_id = item.get("id", "Unknown_ID")
    claims = item.get("claims", {})
    attacks = item.get("attacks", [])
    statements = item.get("statements", {})
    
    print(f"=== Processing Question ID: {q_id} ===")
    
    # Check for malformed or empty data structures
    if not claims or not attacks or not statements:
        print(f"  -> Skipping due to malformed/empty claims, attacks, or statements.\n")
        if statements:
            for stmt_id in statements.keys():
                print(f"  [{stmt_id}]")
                print(f"    Evaluation : null")
                print(f"    Reasoning  : null\n")
        else:
            print("  [No statements found]\n")
        continue

    # Evaluate each statement if the graph is well-formed
    for stmt_id, stmt_text in statements.items():
        res = evaluate_verbatim_statement(stmt_text, claims, attacks)
        
        status = "CORRECT" if res.is_correct else "INCORRECT"
        print(f"  [{stmt_id}]")
        print(f"    Evaluation : {status}")
        print(f"    Reasoning  : {res.reasoning}\n")